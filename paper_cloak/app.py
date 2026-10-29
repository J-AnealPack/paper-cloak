"""Webcam/video interface. GUI is intentionally separate from the processing core."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .config import Config
from .pipeline import PaperCloak


def select_page(frame):
    points = []
    name = "Select page: TL TR BR BL, Enter confirms, Backspace undoes, Esc cancels"
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points)<4:
            points.append((x,y))
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name,click)
    while True:
        shown = frame.copy()
        for i,p in enumerate(points):
            cv2.circle(shown,p,5,(0,255,0),-1)
            cv2.putText(shown,str(i+1),p,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        if len(points)>1:
            cv2.polylines(shown,[np.int32(points)],len(points)==4,(0,255,0),2)
        cv2.imshow(name,shown)
        key = cv2.waitKey(20)&255
        if key in (10,13) and len(points)==4:
            cv2.destroyWindow(name)
            return points
        if key in (8,127) and points:
            points.pop()
        if key == 27 or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:
            cv2.destroyWindow(name)
            return None


def panel(raw,result,dynamic):
    def tile(im,label):
        if im is None:
            im = np.zeros((360,480,3),np.uint8)
        if im.ndim==2:
            im = cv2.cvtColor(im.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)
        h,w = im.shape[:2]
        scale = min(480/w,330/h)
        small = cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))))
        canvas = np.zeros((360,480,3),np.uint8)
        y,x = 30+(330-small.shape[0])//2,(480-small.shape[1])//2
        canvas[y:y+small.shape[0],x:x+small.shape[1]] = small
        cv2.putText(canvas,label,(10,21),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        return canvas
    t = result.tracking
    state = f"TRACKING {t.count} pts | " if t.ok else "LOST | "
    state += "updates ON" if dynamic else "updates OFF"
    if dynamic and result.committed is not None:
        state += f" | learn {100*result.committed.mean():.0f}%"
    return np.vstack([np.hstack([tile(raw,"Raw camera"),tile(result.output,state)]),
                      np.hstack([tile(result.clean,"Clean page memory"),tile(result.blocked,"Blocked update regions (white)")])])


def main():
    p = argparse.ArgumentParser(description="Paper Cloak: a see-through document camera")
    p.add_argument("--source",default="0",help="Camera index or video path")
    p.add_argument("--config")
    p.add_argument("--model",choices=["similarity","homography"])
    p.add_argument("--dynamic",action="store_true",help="Enable experimental white-paper updates")
    p.add_argument("--corners",help="JSON file containing [[x,y],...] in TL TR BR BL order")
    p.add_argument("--headless",action="store_true")
    p.add_argument("--output",help="Output video path (.avi recommended)")
    p.add_argument("--max-frames",type=int,default=0)
    args = p.parse_args()
    if args.headless and not args.corners:
        p.error("--headless requires --corners")
    if args.headless and not args.output:
        p.error("--headless requires --output")
    cfg = Config.load(args.config)
    if args.model: cfg.model = args.model
    if args.dynamic: cfg.dynamic = True
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    writer = None
    try:
        ok,frame = cap.read()
        if not ok: raise RuntimeError("Could not open camera/video. Try --source 1 or check camera permissions.")
        if args.corners:
            with open(args.corners,encoding="utf-8") as f: corners = json.load(f)
        else:
            print("Place an unobstructed page in view. Press SPACE to freeze and select its corners.")
            while True:
                cv2.imshow("Preview: SPACE selects, Esc quits",frame)
                k = cv2.waitKey(20)&255
                if k==27: return
                if k==32: break
                ok,frame = cap.read()
                if not ok: return
            cv2.destroyAllWindows()
            corners = select_page(frame)
        if corners is None: return
        engine = PaperCloak(cfg)
        engine.initialize(frame,corners)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or not 1<=fps<=120: fps=30
        if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            codec = "mp4v" if Path(args.output).suffix.lower()==".mp4" else "MJPG"
            writer = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*codec),fps,(frame.shape[1],frame.shape[0]))
            if not writer.isOpened(): raise RuntimeError("Could not open video output")
        count = 0
        print("D: toggle updates | R: reselect CLEAN page | S: save clean_page.png | Q/Esc: quit")
        while True:
            result = engine.process(frame)
            if writer: writer.write(result.output)
            count += 1
            if not args.headless:
                cv2.imshow("Paper Cloak",panel(frame,result,cfg.dynamic))
                key = cv2.waitKey(max(1,int(1000/fps)) if isinstance(source,str) else 1)&255
                if key in (27,ord('q')): break
                if key==ord('d'): cfg.dynamic = not cfg.dynamic
                if key==ord('s'):
                    cv2.imwrite("clean_page.png",result.clean)
                    print("Saved clean_page.png")
                if key==ord('r'):
                    points = select_page(frame)
                    if points is not None:
                        try: engine.initialize(frame,points)
                        except ValueError as exc: print(exc)
                if cv2.getWindowProperty("Paper Cloak",cv2.WND_PROP_VISIBLE)<1: break
            if args.max_frames and count>=args.max_frames: break
            ok,frame = cap.read()
            if not ok: break
        print(f"Processed {count} frames")
    finally:
        cap.release()
        if writer: writer.release()
        if not args.headless: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
