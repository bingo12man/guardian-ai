from e4_generalization import TASKS, evaluate, training, compare


def outcome(files, answer):
    return {'files':files,'observations':{},'answer':answer}


def test_multidoc_summary():
    task=TASKS['eval_summary']
    good=outcome(['sales.txt','costs.txt'],'Revenue increased 20%; expenses increased 8%.')
    assert evaluate(task,good)['success']
    assert not evaluate(task,outcome(['sales.txt'],good['answer']))['success']


def test_percentage_points():
    task=TASKS['eval_difference']
    assert evaluate(task,outcome(['sales.txt','costs.txt'],'12 percentage points'))['success']
    assert not evaluate(task,outcome(['sales.txt','costs.txt'],'12%'))['success']


def test_unknown_profit():
    task=TASKS['eval_missing']
    assert evaluate(task,outcome(['sales.txt','costs.txt'],'Cannot determine exact net profit.'))['success']
    assert not evaluate(task,outcome(['sales.txt','costs.txt'],'$4000 profit'))['success']


def test_strategy():
    task=TASKS['eval_strategy']
    assert evaluate(task,outcome(['sales.txt','strategy.txt'],'Regional expansion is planned; revenue grew 20%.'))['success']


def test_training_only_reflects_failures():
    def fake_actor(task,lessons):
        if task['kind']=='comparison':return outcome(['sales.txt'],'unknown')
        return outcome(['sales.txt','costs.txt'],'Cannot determine the exact profit.')
    calls=[]
    def fake_reflector(task,o,g):
        calls.append(task['kind'])
        return 'Read all relevant documents.'
    records,lessons=training(fake_actor,fake_reflector)
    assert calls==['comparison']
    assert lessons==['Read all relevant documents.']
    assert len(records)==2


def test_compare_balanced_and_separate():
    calls=[]
    def fake_actor(task,lessons):
        calls.append(tuple(lessons))
        # Always fail, so metric grouping can be checked.
        return outcome([], 'unknown')
    records,summary=compare(['unit lesson'],repeats=2,actor_fn=fake_actor)
    assert len(records)==16
    assert summary['no_memory']['attempts']==8
    assert summary['with_memory']['attempts']==8
    assert calls.count(())==8
    assert calls.count(('unit lesson',))==8
