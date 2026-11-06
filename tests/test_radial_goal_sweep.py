"""Multiple goal references, matched cohorts and dynamic immutable artifacts."""
import numpy as np
import pytest
import torch

from test_radial_ranking import prepared_radial
from test_radial_goal_transfer import transfer_source
from src.mylonas_ijepa_wm import radial_goal_transfer as transfer


GOALS=[[55.656854,42],[54,46],[50,47.656854],[46,46],
       [44.343146,42],[46,38],[50,36.343146],[54,38]]


def test_nine_references_cross_encoder_batch_without_duplicate_encodes(transfer_source):
    source,out,env,encode,signature=transfer_source
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    seen=[]
    def counted(frames):
        seen.extend(f.tobytes() for f in frames)
        return encode(frames)+len(frames)*1e-4
    r=transfer.evaluate(source,out,2,GOALS,env,counted,signature,render=False)
    assert len(r['goals'])==9 and len(seen)==410 and len(set(seen))==410
    assert [g['goal'] for g in r['goals']]==[[50.,42.]]+GOALS
    assert all(g['raw_self_cost']==0 and g['adapted_self_cost']<1e-8 for g in r['goals'])
    assert len({tuple(g['common_mask']) for g in r['goals']})==1
    assert len({g['heldout_common']['count'] for g in r['goals']})==1
    old_id=np.flatnonzero(np.all(np.asarray(r['points'])==[50,42],axis=1)).item()
    assert r['goals'][6]['raw_costs'][old_id]==pytest.approx(8,rel=1e-5)
    assert (out/'goal_8.png').is_file()
    targets=torch.load(out/'goals.pt',weights_only=True)
    assert len(targets['tokens'])==9 and targets['positions']==[[50.,42.]]+GOALS
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    def forbidden(frames):raise AssertionError('completed sweep encoded again')
    assert len(transfer.evaluate(source,out,2,GOALS,env,forbidden,signature,render=False)['goals'])==9
    with pytest.raises(ValueError,match='inputs|config'):
        transfer.evaluate(source,out,2,list(reversed(GOALS)),env,forbidden,signature,render=False)
    frame=out/'goal_8.png';frame.write_bytes(frame.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='artifact|hash'):
        transfer.load_report(source,out,2,GOALS,render=False)


def test_invalid_goal_lists_reject_before_encoding(transfer_source):
    source,out,env,_,signature=transfer_source
    def forbidden(frames):raise AssertionError('invalid goals reached encoder')
    for goals in ([],[[46,38],[46,38]],[[50,42],[46,38]],[[46,float('nan')]],[[46,38,1]],[[50,8]]):
        with pytest.raises(ValueError,match='goal|region|radius'):
            transfer.evaluate(source,out,2,goals,env,forbidden,signature,render=False)


def test_empty_all_goal_intersection_remains_na_and_summary_renders(transfer_source):
    source,out,env,encode,signature=transfer_source
    r=transfer.evaluate(source,out,1,GOALS,env,encode,signature,render=False)
    assert all(g['heldout_common']['count']==0 for g in r['goals'])
    assert all(g['heldout_common']['adapted']['mae'] is None for g in r['goals'])
    from src.mylonas_ijepa_wm.radial_goal_transfer_plots import plot_goal_summary
    plot_goal_summary(out,r)
    assert (out/'goal_summary.png').is_file() and (out/'goal_summary.pdf').is_file()


def test_later_goal_missing_figure_regenerates_from_verified_report(transfer_source):
    source,out,env,encode,signature=transfer_source
    selected=GOALS[:2]
    transfer.evaluate(source,out,2,selected,env,encode,signature,render=True)
    path=out/'goal_2_heatmaps.png';path.unlink()
    transfer.load_report(source,out,2,selected)
    assert path.is_file() and (out/'goal_summary.png').is_file()


def test_notebook_empty_goal_list_rejects_before_cli_fallback():
    import ast,json,sys
    from pathlib import Path
    notebook=Path(__file__).resolve().parents[1]/'notebooks/radial_goal_sweep_colab.ipynb'
    n=json.loads(notebook.read_text())
    tree=ast.parse(''.join(n['cells'][3]['source']))
    commands=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='command']
    context=dict(sys=sys,SOURCE_RUN=Path('source'),OUTPUT=Path('output'),SOURCE_STAGE=4,
        NEW_GOALS=[],ENCODER=Path('encoder'),DINO_ROOT=Path('renderer'))
    exec(compile(ast.Module(body=commands,type_ignores=[]),'notebook command','exec'),context)
    with pytest.raises(ValueError,match='NEW_GOALS|goal'):
        context['command']('evaluate')
    with pytest.raises(ValueError,match='NEW_GOALS|goal'):
        context['command']('report')
