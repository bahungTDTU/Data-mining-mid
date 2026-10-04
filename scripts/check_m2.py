"""Write concrete validation evidence; PostgreSQL must be running."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'commands':[], 'status':'running'}
    commands = [([sys.executable,'-m','pip','check'],ROOT),
                ([sys.executable,'-m','alembic','-c','backend/alembic.ini','check'],ROOT),
                ([sys.executable,'-m','pytest','tests','backend/tests','-q'],ROOT),
                (['npm.cmd','run','build'],ROOT/'frontend'),
                ([sys.executable,'-m','scripts.verify_m1'],ROOT)]
    try:
        for args, cwd in commands:
            result = subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            report['commands'].append({'args':args[1:] if args[0]==sys.executable else args, 'cwd':str(cwd.relative_to(ROOT)),
                                       'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:
                raise RuntimeError('Check failed: ' + ' '.join(args[1:]))
        report['status']='passed'
    except Exception as exc:
        report['status']='failed'
        report['error']=str(exc)
        raise
    finally:
        folder=ROOT/'experiments/m2'; folder.mkdir(parents=True,exist_ok=True)
        (folder/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('M2 checks passed: dependencies, migrations, tests, web build, M1 artifact audit.')


if __name__=='__main__':
    main()
