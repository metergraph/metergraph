import json
from dataclasses import asdict
from pathlib import Path
import pytest
from metergraph_core.reliability import classify_model_reliability

CASES = json.loads((Path(__file__).parent / 'fixtures/reliability.json').read_text())

@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_conformance(case):
    assert asdict(classify_model_reliability(**case['input'])) == case['expected']

@pytest.mark.parametrize('kwargs,expected', [
    ({'provider':'meta','http_status_code':403,'error':{'error':{'code':'user_blocked'}}}, ('excluded',None)),
    ({'error':{'type':'APIConnectionError'}}, (None,None)),
    ({'response':{'choices':[{'finish_reason':'stop','message':{'content':'ok'}},{}]}}, ('available',None)),
    ({'response':{'status':'completed','output':[{}]}}, (None,None)),
    ({'response':{'candidates':[{'finishReason':'STOP'}]}}, ('available',None)),
    ({'response':{'promptFeedback':{'blockReason':'NEW_UNKNOWN'}}}, (None,None)),
    ({'finish_reason':'stop'}, ('available',None)),
    ({'error':{'status':429}}, ('unavailable',None)),
])
def test_sparse_and_error_boundaries(kwargs, expected):
    result = classify_model_reliability(**kwargs)
    assert (result.availability,result.refused) == expected


def test_native_objects_match_mappings():
    from types import SimpleNamespace
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason='stop', message=SimpleNamespace(content=None, refusal=None))])
    result = classify_model_reliability(response=response)
    assert result.availability == 'available'
    assert result.refused is False


def test_stream_refusal_survives_terminal_usage_and_error():
    chunks = [{'choices':[{'delta':{'refusal':'policy'},'finish_reason':None}]},
              {'choices':[{'delta':{},'finish_reason':'stop'}]}, {'usage':{'total_tokens':2}}]
    result = classify_model_reliability(response=chunks[-1], stream_chunks=chunks,
                                       stream_complete=True)
    assert (result.availability, result.refused) == ('available', True)
    failed = classify_model_reliability(stream_chunks=chunks[:1], error=RuntimeError('cancelled'))
    assert failed.refused is True


def test_cancelled_stream_does_not_infer_non_refusal():
    result = classify_model_reliability(stream_chunks=[{'choices':[{'delta':{'content':'ok'}}]}],
                                       stream_complete=False, error=RuntimeError('cancelled'))
    assert result.refused is None


@pytest.mark.parametrize('chunk', [
    {'type':'message_delta','delta':{'stop_reason':'refusal'}},
    {'type':'content_block_start','content_block':{'type':'refusal','refusal':'policy'}},
    {'type':'response.refusal.delta','delta':'policy'},
    {'type':'response.completed','response':{'status':'completed','output':[{'type':'message','content':[{'type':'refusal','refusal':'policy'}]}]}},
    {'candidates':[{'finishReason':'SAFETY'}]},
])
def test_native_stream_refusals(chunk):
    result = classify_model_reliability(stream_chunks=[chunk, {'usage':{}}], stream_complete=True)
    assert (result.availability, result.refused) == ('available', True)


def test_normal_stream_terminal_has_unknown_refusal_coverage():
    result = classify_model_reliability(stream_chunks=[
        {'choices':[{'delta':{'content':'ok'}, 'finish_reason':None}]},
        {'choices':[{'delta':{}, 'finish_reason':'stop'}]},
        {'usage':{}}], stream_complete=True)
    assert (result.availability, result.refused) == ('available', None)
