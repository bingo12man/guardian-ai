from e3_reflexion import attempt, make_lesson, compare


def bad_actor(question, lessons):
    if 'expenses' in question:
        filename = 'costs.txt' if lessons else 'sales.txt'
        answer = '8%' if lessons else 'unknown'
    else:
        filename, answer = 'sales.txt', '20%'
    return {'filename': filename, 'answer': answer,
            'observation': {'status': 'success', 'content': answer}}


def fake_reflector(question, outcome, evaluation):
    return 'Choose the document relevant to the question and preserve the percent unit.'


def test_attempt_grades_each_task():
    assert not attempt('expense', [], bad_actor)['evaluation']['success']
    assert attempt('revenue', [], bad_actor)['evaluation']['success']


def test_training_generates_lesson_only_on_failure():
    trained = make_lesson('expense', bad_actor, fake_reflector)
    assert trained['lesson'].startswith('Choose')
    success = make_lesson('revenue', bad_actor, fake_reflector)
    assert success['lesson'] == ''


def test_comparison_balances_conditions():
    result = compare(['expense', 'revenue'], ['lesson'], repeats=2, actor=bad_actor)
    assert result['summary']['no_memory']['attempts'] == 4
    assert result['summary']['with_memory']['attempts'] == 4
    assert result['summary']['with_memory']['successful'] == 4
    assert result['summary']['no_memory']['successful'] == 2


def test_empty_lesson_does_not_fake_improvement():
    result = compare(['expense'], [], repeats=2, actor=bad_actor)
    assert result['summary']['no_memory']['successful'] == 0
    assert result['summary']['with_memory']['successful'] == 0
