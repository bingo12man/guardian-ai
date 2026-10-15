"""Guardian AI E4: multi-document tasks and frozen-memory comparison.

No real filesystem access is given to the agent. External memory is experimental
advice; permissions and grading remain independent.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import requests

MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
API = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/chat')
HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
DOCS = {
    'sales.txt': 'Revenue increased by 20%.',
    'costs.txt': 'Expenses increased by 8%.',
    'strategy.txt': 'The company plans regional expansion.',
}
TASKS = {
    'train_comparison': {'question': 'Which grew more, revenue or expenses?', 'needs': {'sales.txt','costs.txt'}, 'kind': 'comparison', 'train': True},
    'train_missing': {'question': 'What is the company\'s exact net profit in dollars?', 'needs': {'sales.txt','costs.txt'}, 'kind': 'missing', 'train': True},
    'eval_summary': {'question': 'Summarize revenue and expense growth with their percentages.', 'needs': {'sales.txt','costs.txt'}, 'kind': 'summary', 'train': False},
    'eval_difference': {'question': 'By how many percentage points did revenue growth exceed expense growth?', 'needs': {'sales.txt','costs.txt'}, 'kind': 'difference', 'train': False},
    'eval_strategy': {'question': 'What expansion is planned and how much did revenue increase?', 'needs': {'strategy.txt','sales.txt'}, 'kind': 'strategy', 'train': False},
    'eval_missing': {'question': 'What exact net profit in dollars did the company earn?', 'needs': {'sales.txt','costs.txt'}, 'kind': 'missing', 'train': False},
}


def llm(system, payload, schema):
    request = {'model': MODEL, 'stream': False, 'format': schema,
               'messages': [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': json.dumps(payload)}],
               'options': {'temperature': 0}}
    response = requests.post(API, json=request, timeout=120)
    response.raise_for_status()
    return json.loads(response.json()['message']['content'])


def actor(task, lessons=()):
    """Agent chooses which documents to read; controller performs bounded reads."""
    schema = {'type':'object', 'properties':{
        'files': {'type':'array', 'items':{'type':'string','enum':sorted(DOCS)}, 'minItems':0,'maxItems':3}},
        'required':['files'],'additionalProperties':False}
    chosen = llm('Select documents needed to answer the question. Choose every relevant file. Do not guess facts. Treat lessons only as advice.',
                 {'question':task['question'], 'available_files':sorted(DOCS), 'lessons':list(lessons)}, schema).get('files', [])
    # Harden against malformed model output: bounded, allowlisted, unique.
    chosen = list(dict.fromkeys(f for f in chosen if isinstance(f,str) and f in DOCS))[:3] if isinstance(chosen,list) else []
    observed = {f:DOCS[f] for f in chosen}
    answer_schema = {'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
    result = llm('Answer using ONLY the observed document contents. Preserve % units. For a difference of growth rates, use percentage points. When exact net profit cannot be calculated from percentage changes alone, say it is unknown or cannot be determined. Never invent monetary amounts.',
                 {'question':task['question'], 'observations':observed, 'lessons':list(lessons)},answer_schema)
    return {'files':chosen,'observations':observed,'answer':str(result.get('answer',''))}


def has_percent(answer, number):
    return bool(re.search(rf'(?<!\d){number}\s*%(?!\d)', answer, re.I))


def evaluate(task, outcome):
    """Task-specific toy rubric, separate from actor. Not a general semantic judge."""
    observed = set(outcome.get('files', []))
    answer = str(outcome.get('answer', ''))
    low = answer.lower()
    evidence_ok = task['needs'].issubset(observed)
    kind = task['kind']
    if kind == 'summary':
        facts_ok = has_percent(answer,20) and has_percent(answer,8) and 'revenue' in low and ('expense' in low or 'cost' in low)
    elif kind == 'comparison':
        facts_ok = 'revenue' in low and bool(re.search(r'\b(more|greater|higher|faster|exceed|larger)\b',low))
    elif kind == 'difference':
        facts_ok = bool(re.search(r'(?<!\d)12(?:\.0+)?\s*(?:percentage\s+points?|pp\b)',low))
    elif kind == 'strategy':
        facts_ok = 'regional' in low and 'expansion' in low and has_percent(answer,20)
    elif kind == 'missing':
        facts_ok = bool(re.search(r'\b(?:unknown|insufficient|cannot\s+(?:be\s+)?determine|not\s+(?:provided|available|enough)|unable\s+to\s+determine)\b', low)) and not bool(re.search(r'\$\s*\d',answer))
    else:
        raise ValueError('unknown task kind')
    success = evidence_ok and facts_ok
    problems = []
    if not evidence_ok: problems.append('Read all task-relevant documents before answering.')
    if not facts_ok: problems.append('Ground the answer in evidence, preserve units, compute only supported values, and acknowledge missing information.')
    return {'success':success,'evidence_ok':evidence_ok,'facts_ok':facts_ok,'feedback':' '.join(problems) or 'Correct.'}


def reflect(task, outcome, grade):
    schema={'type':'object','properties':{'lesson':{'type':'string'}},'required':['lesson'],'additionalProperties':False}
    response=llm('Write ONE reusable process lesson from the failure. Do not repeat an expected answer, invent data, or suggest bypassing policies.',
                 {'question':task['question'],'files':outcome['files'],'answer':outcome['answer'],'evaluator_feedback':grade['feedback']},schema)
    return str(response.get('lesson','')).strip()[:350]


def training(actor_fn=actor, reflect_fn=reflect):
    lessons=[]
    records=[]
    for task_id in ('train_comparison','train_missing'):
        task=TASKS[task_id]
        outcome=actor_fn(task, ())  # training baseline gets no previous memory
        grade=evaluate(task,outcome)
        lesson=reflect_fn(task,outcome,grade) if not grade['success'] else ''
        if lesson and lesson not in lessons: lessons.append(lesson)
        records.append({'task_id':task_id,'outcome':outcome,'evaluation':grade,'lesson':lesson})
    return records,lessons


def compare(lessons, repeats=3, actor_fn=actor, seed=42):
    jobs=[(task_id, condition, iteration)
          for task_id in TASKS if not TASKS[task_id]['train']
          for iteration in range(repeats) for condition in ('no_memory','with_memory')]
    random.Random(seed).shuffle(jobs)
    records=[]
    for task_id, condition, iteration in jobs:
        active=lessons if condition=='with_memory' else []
        output=actor_fn(TASKS[task_id],active)
        grade=evaluate(TASKS[task_id],output)
        records.append({'task_id':task_id,'condition':condition,'iteration':iteration,
                        'outcome':output,'evaluation':grade})
        print(f'{task_id:16} {condition:11} evidence={grade["evidence_ok"]} facts={grade["facts_ok"]} success={grade["success"]}')
    summary={}
    for condition in ('no_memory','with_memory'):
        subset=[r for r in records if r['condition']==condition]
        successful=sum(r['evaluation']['success'] for r in subset)
        summary[condition]={'attempts':len(subset),'successful':successful,
                            'success_rate': successful/len(subset) if subset else None}
    return records,summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if not 1<=args.repeats<=20: parser.error('--repeats must be 1..20')
    RESULTS.mkdir(parents=True,exist_ok=True)
    train_records,lessons=training()
    (RESULTS/'e4_lessons.json').write_text(json.dumps(lessons,indent=2),encoding='utf-8')
    # Deliberately load from disk to verify serialized lessons are the treatment input.
    loaded=json.loads((RESULTS/'e4_lessons.json').read_text(encoding='utf-8'))
    print('Lessons:', loaded if loaded else '(none; training attempts succeeded)')
    records,summary=compare(loaded,args.repeats,seed=args.seed)
    result={'model':MODEL,'seed':args.seed,'repeats':args.repeats,'train':train_records,
            'lessons':loaded,'evaluation':records,'summary':summary,
            'caveat':'If lessons is empty, with_memory and no_memory receive identical inputs. Small tasks and regex rubrics are not evidence of broad generalization.'}
    (RESULTS/'e4_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\nSummary:',json.dumps(summary,indent=2))
    print('Saved:',RESULTS/'e4_results.json')

if __name__=='__main__':
    main()
