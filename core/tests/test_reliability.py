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


@pytest.mark.parametrize('row,expected', [
    ({'status_code':'error','error':True,'otel':{'finish_reasons':'[invalid'}}, (None,None,None)),
    ({'otel':{'finish_reasons':'["stop","content_filter"]'}}, (None,'available',True)),
    ({'response_received':True,'error_type':'APITimeoutError'}, (None,'unavailable',None)),
    ({'http_status_code':400,'error_code':'content_policy_violation','refusal_signal':False}, (400,'available',True)),
    ({'finish_reason_raw':'safety','otel':{'finish_reasons':['stop']},'refusal_signal':False}, (None,'available',True)),
    ({'response_received':True,'availability':'unavailable'}, (None,'available',None)),
    ({'response_received':True,'refusal_signal':False}, (None,'available',False)),
    ({'refusal_signal':False}, (None,None,None)),
    ({'refusal_signal':True}, (None,None,None)),
    ({'http_status_code':429,'refusal_signal':True,'response_received':True}, (429,'unavailable',None)),
    ({'http_status_code':503,'refusal_signal':True}, (503,'unavailable',None)),
    ({'http_status_code':429,'availability':'unavailable','refused':True}, (429,None,None)),
    ({'http_status_code':403,'refused':True}, (403,'available',True)),
])
def test_recorded_call_evidence(row, expected):
    from metergraph_core import classify_recorded_call
    result = classify_recorded_call(row)
    assert (result.http_status_code,result.availability,result.refused) == expected
