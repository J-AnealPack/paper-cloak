"""Run from the project root: python generate_tests.py
Generates deterministic procedural hand/perspective tests, not real-hand footage.
Requires the existing paper_cloak package plus numpy and OpenCV.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from paper_cloak import Config, PaperCloak
from paper_cloak.app import panel
from paper_cloak.geometry import project

cv2.setNumThreads(2)
W,H,FPS,N=800,600,24,240
CORNERS=np.float32([[0,0],[359,0],[359,479],[0,479]])

def paper(i,writing):
    p=np.full((480,360,3),247,np.uint8)
    cv2.rectangle(p,(6,6),(353,473),(40,40,40),1)
    rng=np.random.default_rng(27)
    for y in range(20,460,32):
        for x in (15,321):
            # Asymmetric border features resist repetitive-pattern confusion.
            for k in range(4):
                a,b=rng.integers(0,15,2)
                cv2.rectangle(p,(x+int(a),y+int(b)),(x+int(a)+3,y+int(b)+4),(20,20,20),-1)
    cv2.putText(p,'PAPER CLOAK',(57,58),0,.72,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(p,'Tracking + clean memory',(47,90),0,.48,(30,30,30),1,cv2.LINE_AA)
    for j,s in enumerate(['1. Keep border details visible','2. Cover the center of the page','3. Watch the clean view']):
        cv2.putText(p,s,(43,133+j*25),0,.43,(45,45,45),1,cv2.LINE_AA)
    cv2.line(p,(45,201),(315,201),(100,100,100),1)
    if writing and i>=75:
        # Growing stroke: each frame extends the visible path, like writing.
        pts=np.array([[65,278],[83,244],[98,278],[114,245],[131,278],[148,245],[165,277],[185,249],[204,277],[225,245],[247,277],[276,248]],np.int32)
        progress=min(1,(i-75)/85)
        length=progress*(len(pts)-1); k=int(length)
        active=list(pts[:k+1])
        if k<len(pts)-1:active.append(np.int32(pts[k]+(pts[k+1]-pts[k])*(length-k)))
        if len(active)>1:cv2.polylines(p,[np.array(active)],False,(15,15,15),1,cv2.LINE_AA)
    if writing and i>=160:
        cv2.putText(p,'NEW TEXT',(75,330),0,.8,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(p,'Planar test sheet',(105,423),0,.5,(50,50,50),1,cv2.LINE_AA)
    return p

def page_H(i,kind):
    t=i/(N-1); w=2*np.pi*t
    tilt=kind in ('perspective','writing','stress')
    ax=np.deg2rad(20+15*np.sin(w)) if tilt else 0
    ay=np.deg2rad(-26+25*np.sin(w*.8)) if tilt else 0
    az=np.deg2rad(12*np.sin(w))
    Rx=np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry=np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz=np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    pts=np.column_stack([CORNERS-[180,240],np.zeros(4)]) @ (Rz@Ry@Rx).T
    pts[:,2]+=850+35*np.sin(w)
    dest=pts[:,:2]*800/pts[:,2:]+[400+22*np.sin(w),300]
    return cv2.getPerspectiveTransform(CORNERS,dest.astype(np.float32))

def hand_layer(i,kind):
    """Articulating five-finger silhouette rendered independently of page motion."""
    t=i/(N-1)
    mask=np.zeros((600,450),np.uint8)
    cv2.ellipse(mask,(223,320),(79,105),-5,0,360,255,-1)
    cv2.rectangle(mask,(170,365),(278,599),255,-1)
    spread=10*np.sin(i*.07)
    for j,(x,y) in enumerate([(162,250),(202,215),(244,208),(283,242)]):
        tip=(int(x+(j-1.5)*spread),int(y-102-12*np.sin(i*.08+j)))
        cv2.line(mask,(x,310),tip,255,29,cv2.LINE_AA)
        cv2.circle(mask,tip,14,255,-1,cv2.LINE_AA)
    cv2.line(mask,(173,330),(108,int(272+12*np.sin(i*.06))),255,35,cv2.LINE_AA)
    # Enter and leave from below, with a pause during the middle of the clip.
    enter=np.clip((t-.10)/.17,0,1);leave=np.clip((t-.78)/.16,0,1)
    visibility=min(enter,1-leave)
    scale=.90 if kind!='stress' else 1.65
    center=(400+78*np.sin(t*2*np.pi),320+(1-visibility)*510)
    if kind=='writing':center=(350+125*np.clip((i-75)/85,0,1),350+(1-visibility)*510)
    A=cv2.getRotationMatrix2D((223,310),15*np.sin(t*2*np.pi),scale)
    A[:,2]+=np.array(center)-[223,310]
    alpha=cv2.warpAffine(mask,A,(W,H)).astype(np.float32)/255
    # Color gradient and crease details, no photographic realism claimed.
    yy,xx=np.mgrid[:H,:W]
    skin=np.empty((H,W,3),np.uint8)
    for ch,b in enumerate((100,155,210)):
        skin[:,:,ch]=np.clip(b+8*np.sin(xx/70)+5*np.cos(yy/60),0,255)
    return alpha,skin

def sequence(kind):
    for i in range(N):
        P=paper(i,kind=='writing'); transform=page_H(i,kind)
        grid=np.full((H,W,3),(78,71,65),np.uint8)
        for y in range(0,H,40):cv2.line(grid,(0,y),(W,y),(86,79,73),1)
        warp=cv2.warpPerspective(P,transform,(W,H))
        pm=cv2.warpPerspective(np.full(P.shape[:2],255,np.uint8),transform,(W,H))>0
        truth=grid.copy();truth[pm]=warp[pm]
        alpha,skin=hand_layer(i,kind)
        if kind=='perspective':alpha*=0  # isolate camera/page geometry
        shadow=cv2.GaussianBlur(alpha,(21,21),5)*.16
        raw=truth.astype(float)*(1-shadow[:,:,None])
        raw=(raw*(1-alpha[:,:,None])+skin*alpha[:,:,None]).astype(np.uint8)
        yield raw,truth,transform,pm,alpha

def build(kind,root):
    out=root/kind;out.mkdir(parents=True,exist_ok=True)
    cfg=Config(page_width=360,model='homography' if kind!='hand_sweep' else 'similarity',dynamic=kind=='writing',confirmation_frames=5)
    cfg.save(out/'config.json')
    engine=PaperCloak(cfg)
    writers=[]
    for name,size in [('input.mp4',(W,H)),('comparison.mp4',(960,720))]:
        v=cv2.VideoWriter(str(out/name),cv2.VideoWriter_fourcc(*'mp4v'),FPS,size)
        if not v.isOpened():raise RuntimeError('mp4v unavailable; install desktop OpenCV')
        writers.append(v)
    errors=[];lost=[];coverage=[]
    try:
        for i,(raw,truth,T,pm,alpha) in enumerate(sequence(kind)):
            corners=project(CORNERS,T)
            if i==0:
                engine.initialize(raw,corners)
                (out/'corners.json').write_text(json.dumps(corners.tolist()))
            res=engine.process(raw)
            if res.tracking.ok:errors.append(float(np.linalg.norm(project(engine.canonical,res.tracking.H)-corners,axis=1).mean()))
            else:lost.append(i)
            coverage.append(float(((alpha>.5)&pm).sum()/pm.sum()))
            dashboard=panel(raw,res,cfg.dynamic)
            writers[0].write(raw);writers[1].write(dashboard)
            if i==N//2:cv2.imwrite(str(out/'preview.jpg'),dashboard)
        report={'case':kind,'frames':N,'fps':FPS,'tracked_frames':N-len(lost),'first_lost_frame':lost[0] if lost else None,'mean_corner_error_px_while_tracking':float(np.mean(errors)) if errors else None,'max_page_occlusion_fraction':max(coverage),'synthetic':True}
        if kind=='writing':
            # Score actual ink pixels; whole-page MAE hides missing thin text.
            target=np.zeros((480,360),np.uint8)
            target[220:365,40:305]=255
            canonical_map=np.linalg.inv(res.tracking.H) @ T if res.tracking.ok else None
            if canonical_map is not None:
                expected=cv2.warpPerspective(paper(N-1,True),canonical_map,engine.size)
                region=cv2.warpPerspective(target,canonical_map,engine.size)>0
                ink=(cv2.cvtColor(expected,cv2.COLOR_BGR2GRAY)<145)&region
                recovered=cv2.dilate((cv2.cvtColor(res.clean,cv2.COLOR_BGR2GRAY)<165).astype(np.uint8),np.ones((3,3),np.uint8))>0
                report['new_ink_recall_1px_tolerance']=float(recovered[ink].mean())
            else:
                report['new_ink_recall_1px_tolerance']=None
        cv2.imwrite(str(out/'final_clean.png'),res.clean)
        cv2.imwrite(str(out/'final_comparison.jpg'),dashboard)
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return report
    finally:
        for v in writers:v.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='test_cases');p.add_argument('--case',choices=['all','hand_sweep','perspective','writing','stress'],default='all');args=p.parse_args()
    root=Path(args.output)
    kinds=['hand_sweep','perspective','writing','stress'] if args.case=='all' else [args.case]
    reports=[build(k,root) for k in kinds]
    (root/'summary.json').write_text(json.dumps(reports,indent=2))
