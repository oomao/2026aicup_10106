import numpy as np, pandas as pd
from pathlib import Path
R = Path("E:/AICUP_O")
ta = np.load(R/"models/v1080_transformer_transductive/outputs/test_action.npy")
tu = np.load(R/"models/v1080_transformer_transductive/outputs/test_rally_uids.npy")
o=np.argsort(tu); tu=tu[o]; ta=ta[o]; tf=ta.argmax(1); conf=ta.max(1)
v701 = pd.read_csv(R/"_NEW_PUBLIC/result/lb_history/0.4141329_v701_withinmatch_point/sub_v701_withinmatch_point_overlay.csv").sort_values("rally_uid").reset_index(drop=True)
assert (v701.rally_uid.values==tu).all()
prod=v701.actionId.values
train=pd.read_csv(R/"data/train.csv"); test=pd.read_csv(R/"data/test.csv")
trpl=set(train.gamePlayerId.unique())|set(train.gamePlayerOtherId.unique())
t1=test[test.strikeNumber==1].drop_duplicates("rally_uid").set_index("rally_uid")
oov=np.array([(t1.loc[u,"gamePlayerId"] not in trpl) or (t1.loc[u,"gamePlayerOtherId"] not in trpl) for u in tu])

def gates(name, act):
    a0=(act==0).mean(); p0=(v701.pointId.values==0).mean()  # point unchanged (v701)
    # enforce action0 -> point0 constraint check
    chg=(act!=prod).sum()
    print(f"[{name}] cells_changed={chg} action0_rate={a0:.4f} point0_rate(v701)={p0:.4f}")
    return a0

OUT=R/"result/staging_day48"; OUT.mkdir(parents=True, exist_ok=True)
def build(name, act):
    pt=v701.pointId.values.copy()
    pt[act==0]=0  # action0->point0 constraint
    sub=pd.DataFrame({"rally_uid":tu,"actionId":act.astype(int),"pointId":pt.astype(int),"serverGetPoint":v701.serverGetPoint.values})
    sub.to_csv(OUT/f"{name}.csv",index=False); return sub

# FULL swap
full=tf.copy(); gates("FULL_swap", full); build("sub_day48_v1080_FULL_swap", full)
# OOV-gated (transformer only on OOV rallies)
g_oov=np.where(oov, tf, prod); gates("OOV_gated", g_oov); build("sub_day48_v1080_OOVgated", g_oov)
# OOV + confidence>=0.45 gated (tightest)
g_oc=np.where(oov & (conf>=0.45), tf, prod); gates("OOV_conf45", g_oc); build("sub_day48_v1080_OOVconf45", g_oc)
print("\ncells: FULL=%d  OOVgated=%d  OOVconf45=%d"%((full!=prod).sum(),(g_oov!=prod).sum(),(g_oc!=prod).sum()))
print("staged ->", OUT)
