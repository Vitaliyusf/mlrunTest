from mlrun import run_local, RunTemplate, NewTask, mlconf
from os import path
mlconf.dbpath = mlconf.dbpath or './'
out = mlconf.artifact_path or path.abspath('./data')
# {{run.uid}} will be substituted with the run id, so output will be written to different directoried per run
artifact_path = path.join(out, '{{run.uid}}')
task = NewTask(name='demo', params={'p1': 5}, artifact_path=artifact_path).with_secrets('file', 'secrets.txt').set_label('type', 'demo')
# run our task using our new function
run_object = run_local(task, command='training.py')
run_object.uid()
run_object.to_dict()
run_object.state()
run_object.show()
run_object.outputs
run_object.logs()
run_object.artifact('dataset')
