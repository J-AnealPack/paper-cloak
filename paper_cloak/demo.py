"""Deterministic synthetic video with known motion, occlusion and new ink.
Run: python -m paper_cloak.demo --output demo
All outputs are local; no camera or network required.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .config import Config
from .pipeline import PaperCloak
from .geometry import project
from .app import panel


def make_page():
    page = np.full((420,300,3),245,np.uint8)
    # Distributed printed details; central region is reserved for writing.
    for y in range(15,410,35):
        for x in (12,265):
            cv2.rectangle(page,(x,y),(x+15,y+14),(25,25,25),2)
            cv2.line(page,(x,y),(x+15,y+14),(25,25,25),1)
    cv2.putText(page,'PAPER CLOAK',(40,55),cv2.FONT_HERSHEY_SIMPLEX,0.65,(30,30,30),1)
    cv2.putText(page,'Clean page memory',(45,90),cv2.FONT_HERSHEY_SIMPLEX,0.45,(30,30,30),1)
    cv2.line(page,(45,105),(255,105),(70,70,70),1)
    return page


def sequence(count=180, perspective=False):
    page = make_page()
    corners = np.float32([[0,0],[299,0],[299,419],[0,419]])
    for i in range(count):
        t = i/max(count-1,1)
        angle = 18*np.sin(t*2*np.pi)
        scale = 1+0.10*np.sin(t*2*np.pi)
        A = cv2.getRotationMatrix2D((150,210),angle,scale)
        A[:,2] += (170+20*np.sin(t*4*np.pi),30)
        H = np.vstack([A,[0,0,1]])
        if perspective:
            H = H @ np.array([[1,0,0],[0,1,0],[0.00015*np.sin(t*2*np.pi),0,1]])
        truth_page = page.copy()
        if t>0.32:
            cv2.putText(truth_page,'NEW INK',(65,235),cv2.FONT_HERSHEY_SIMPLEX,0.65,(20,20,20),1)
        size=(640,480)
        mask = cv2.warpPerspective(np.full(page.shape[:2],255,np.uint8),H,size)>0
        truth = np.full((480,640,3),60,np.uint8)
        warp = cv2.warpPerspective(truth_page,H,size)
        truth[mask] = warp[mask]
        raw = truth.copy()
        occ = np.zeros(page.shape[:2],np.uint8)
        if 0.15<t<0.65:
            cv2.ellipse(occ,(150,225),(65,45),0,0,360,255,-1)
        cam_occ=cv2.warpPerspective(occ,H,size,flags=cv2.INTER_NEAREST)>0
        raw[cam_occ]=(85,140,195)  # colored occluder, not a realistic hand simulation
        yield raw,truth,project(corners,H),cam_occ,truth_page


def run(output, count=180):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=Config(page_width=300,dynamic=True,confirmation_frames=5)
    cfg.save(out/'demo_config.json')
    engine=PaperCloak(cfg)
    writers=[]
    for name,size in [('input.avi',(640,480)),('output.avi',(640,480)),('comparison.avi',(960,720))]:
        writer=cv2.VideoWriter(str(out/name),cv2.VideoWriter_fourcc(*'MJPG'),30,size)
        if not writer.isOpened(): raise RuntimeError('MJPG encoder unavailable')
        writers.append(writer)
    errors=[];visible=0;raw_errors=[];out_errors=[]
    try:
        for i,(raw,truth,corners,occ,truth_page) in enumerate(sequence(count)):
            if i==0:
                engine.initialize(raw,corners)
                (out/'corners.json').write_text(json.dumps(corners.tolist(),indent=2))
                cv2.imwrite(str(out/'reference_page.png'),make_page())
            result=engine.process(raw)
            if result.tracking.ok:
                visible+=1
                estimate=project(engine.canonical,result.tracking.H)
                errors.append(float(np.mean(np.linalg.norm(estimate-corners,axis=1))))
            if occ.any():
                # Under occlusion the new ink has not been observed; evaluate
                # removal against the last observable base page, not unseen ink.
                base = np.full_like(raw,60)
                H=cv2.getPerspectiveTransform(np.float32([[0,0],[299,0],[299,419],[0,419]]),corners)
                base=cv2.warpPerspective(make_page(),H,(640,480))
                raw_errors.append(float(np.abs(raw.astype(float)-base)[occ].mean()))
                out_errors.append(float(np.abs(result.output.astype(float)-base)[occ].mean()))
            dashboard=panel(raw,result,True)
            for writer,frame in zip(writers,[raw,result.output,dashboard]): writer.write(frame)
            if i==count//2: cv2.imwrite(str(out/'preview.jpg'),dashboard)
        cv2.imwrite(str(out/'final_clean.png'),result.clean)
        # Interior crop excludes permanently protected paper edges.
        final_error=float(np.abs(result.clean.astype(float)-truth_page.astype(float))[110:320,45:255].mean())
        metrics={'frames':count,'tracked_frames':visible,
                 'mean_corner_error_px':float(np.mean(errors)) if errors else None,
                 'max_corner_error_px':float(np.max(errors)) if errors else None,
                 'occluded_region_raw_MAE':float(np.mean(raw_errors)),
                 'occluded_region_output_MAE':float(np.mean(out_errors)),
                 'final_writing_region_MAE':final_error,
                 'note':'Synthetic controlled sequence; not a webcam accuracy estimate.'}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        print(json.dumps(metrics,indent=2))
        return metrics
    finally:
        for writer in writers: writer.release()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='demo')
    parser.add_argument('--frames',type=int,default=180)
    args=parser.parse_args()
    if args.frames<30: parser.error('Use at least 30 frames')
    run(args.output,args.frames)
