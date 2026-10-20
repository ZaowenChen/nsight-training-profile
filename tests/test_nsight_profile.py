import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'skills/nsight-training-profile/scripts/nsight_profile.py'
spec = importlib.util.spec_from_file_location('nsight_profile', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(path):
    with contextlib.closing(sqlite3.connect(path)) as con, con:
        con.executescript('''
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES (1, 'overlapping_kernel'),(2,'copy_api');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                (start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER,
                 streamId INTEGER, correlationId INTEGER, globalPid INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (0,10,0,1,1,1,1),(5,15,0,1,2,2,1),(30,40,0,1,1,3,1),
                (0,100,1,1,1,4,1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
                (start INTEGER,end INTEGER,deviceId INTEGER,bytes INTEGER,copyKind INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (10,20,0,1024,1);
            CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,text TEXT,eventType INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (0,100,'train_step',59);
            CREATE TABLE SAMPLING_EVENTS (timestamp INTEGER);
            CREATE TABLE ENUM_SAMPLING_THREAD_STATE (id INTEGER,value TEXT);
            INSERT INTO ENUM_SAMPLING_THREAD_STATE VALUES (1,'running');
            CREATE TABLE NVTX_PAYLOAD_SCHEMAS (schemaId INTEGER);
            INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1);
        ''')


class SummaryTests(unittest.TestCase):
    def test_union_overlap_memcpy_and_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'trace with spaces.sqlite'
            fixture(db)
            digest = hashlib.sha256(db.read_bytes()).hexdigest()
            report = module.native_summary(db)
            devices = {x['device_id']: x for x in report['devices']}
            self.assertEqual(devices[0]['event_window_ns'], 40)
            self.assertEqual(devices[0]['gpu_activity_union_ns'], 30)
            self.assertEqual(devices[0]['internal_idle_ns'], 10)
            self.assertEqual(devices[0]['largest_internal_gaps'][0]['cause'], 'unattributed')
            self.assertEqual(devices[1]['gpu_activity_union_ns'], 100)
            self.assertEqual(report['nonempty_coverage_tables']['cpu_sampling'], [])
            self.assertEqual(report['nonempty_coverage_tables']['nvtx'], ['NVTX_EVENTS'])
            self.assertEqual(report['top_kernels_all_devices'][0]['name'], 'overlapping_kernel')
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)

    def test_no_gpu_activity_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'cpu.sqlite'
            with contextlib.closing(sqlite3.connect(db)) as con, con:
                con.execute('CREATE TABLE NVTX_EVENTS (start INTEGER)')
            report = module.native_summary(db)
            self.assertEqual(report['status'], 'no_gpu_activity')
            self.assertEqual(report['devices'], [])
            self.assertTrue(report['warnings'])

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'result.txt').write_text('keep')
            with self.assertRaises(ValueError):
                module.new_output(tmp)
            self.assertEqual((Path(tmp) / 'result.txt').read_text(), 'keep')

    def test_capture_preserves_workload_and_handles_missing_native_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'dry run'
            args = types.SimpleNamespace(command=['--','/usr/bin/python3','train.py'],output=str(prefix),
                cpu='auto',duration=10,delay=30,capture_range='time',python_sampling=False,dry_run=True)
            with patch.object(module,'environment',return_value={'native_cpu_sampling_available':False}), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(module.capture(args),0)
            record = json.loads(output.getvalue())
            self.assertIn('--kill=none',record['command'])
            self.assertIn('--sample=none',record['command'])
            self.assertIn('--cpuctxsw=none',record['command'])
            self.assertEqual(record['command'][-2:],['/usr/bin/python3','train.py'])
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_collector_failure_remains_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = module.saved(['/does/not/exist'],Path(tmp)/'result.json',1)
            self.assertIsNone(result['exit_code'])
            self.assertIn('error',result)

    def test_cpu_periodic_samples_exclude_scheduler_and_deduplicate_recursion(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'native-cpu.sqlite'
            with contextlib.closing(sqlite3.connect(db)) as con, con:
                con.executescript('''
                    CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
                    INSERT INTO StringIds VALUES (1,'foo'),(2,'bar'),(3,'module');
                    CREATE TABLE COMPOSITE_EVENTS (id INTEGER PRIMARY KEY,start INTEGER,globalTid INTEGER,cpuCycles INTEGER);
                    INSERT INTO COMPOSITE_EVENTS VALUES (1,100,7,1),(2,200,7,1),(3,300,7,0);
                    CREATE TABLE SAMPLING_CALLCHAINS (id INTEGER,symbol INTEGER,module INTEGER,stackDepth INTEGER);
                    INSERT INTO SAMPLING_CALLCHAINS VALUES
                        (1,1,3,0),(1,1,3,1),(2,2,3,0),(2,1,3,1),(3,1,3,0);
                ''')
            report = module.native_summary(db)['native_cpu_samples']
            self.assertEqual(report['periodic_cpu_samples'],2)
            self.assertEqual(report['other_composite_events'],1)
            self.assertEqual(report['samples_with_callchains'],2)
            own = {r['symbol']:r['samples'] for r in report['self_hotspots']}
            inclusive = {r['symbol']:r['samples'] for r in report['inclusive_hotspots']}
            self.assertEqual(own,{'foo':1,'bar':1})
            self.assertEqual(inclusive,{'foo':2,'bar':1})

    def test_missing_cpu_sampling_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'trace.sqlite'
            fixture(db)
            report = module.native_summary(db)['native_cpu_samples']
            self.assertEqual(report['status'],'not_recorded')
            self.assertIsNone(report['periodic_cpu_samples'])


if __name__ == '__main__':
    unittest.main()
