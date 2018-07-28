import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def objective(x):
    return {'loss': x ** 2,  # This is the same thing as x squared
            'status': STATUS_OK,
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            'attachments': {'time_module': pickle.dumps(time.time)}}


trials = Trials()
best = fmin(objective,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)

msg = trials.trial_attachments(trials.trials[5])['time_module']
time_module = pickle.loads(msg)
