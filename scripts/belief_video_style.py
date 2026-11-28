"""Clean map rendering for saved belief heuristic trajectories."""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BG='#101826';PANEL='#162238';TEXT='#e7edf7';MUTED='#97a9bf'
COLORS=['#61c9ff','#b79aff','#76dfb0','#ffc37b','#f58eb9']
def font(n):return ImageFont.truetype(FONT,n)
def render_frame(a,b,index,config):
    im=Image.new('RGB',(1440,850),BG);draw=ImageDraw.Draw(im)
    for side,data in enumerate((a,b)):
        ix=min(index,len(data['frames'])-1);f=data['frames'][ix];ox=side*720
        title='H1 · Belief-guided search' if side==0 else 'H2 · Two systematic scouts'
        draw.text((ox+30,18),title,font=font(21),fill=TEXT)
        draw.text((ox+30,52),f'Seed {data["summary"]["seed"]}    {f["step"]:4d} s',font=font(15),fill=MUTED)
        draw.text((ox+340,52),f'Removed {f["completed"]}/{data["summary"]["total"]}    Found {f["discovered"]}',font=font(15),fill=TEXT)
        # Draw all spatial layers on a separate image to clip paths at map edges.
        size=660;scale=66
        heat=(np.asarray(f['heat'],float)/255)**1.8
        low=np.array([20,32,49]);high=np.array([87,190,239])
        rgb=(low[None,None,:]+heat[...,None]*(high-low)[None,None,:]).astype('uint8')
        heat_im=Image.fromarray(rgb[::-1]).resize((528,528),Image.Resampling.BILINEAR)
        layer=Image.new('RGB',(size,size),PANEL);layer.paste(heat_im,(66,66))
        d=ImageDraw.Draw(layer)
        def xy(p):return (float(p[0])*scale,size-float(p[1])*scale)
        for k in range(1,10):
            d.line((k*scale,0,k*scale,size),fill='#24364c',width=1)
            d.line((0,k*scale,size,k*scale),fill='#24364c',width=1)
        for p in f.get('polygons',[]):
            assigned=p['assigned'];v=[xy(z) for z in p['vertices']];v.append(v[0])
            d.line(v,fill=COLORS[assigned%len(COLORS)] if assigned>=0 else '#51677e',width=2 if assigned>=0 else 1)
        for i,pos in enumerate(f['pos']):
            trail=[xy(frame['pos'][i]) for frame in data['frames'][max(0,ix-24):ix+1]]
            if len(trail)>1:d.line(trail,fill=COLORS[i%len(COLORS)],width=2)
            if not f['active'][i]:continue
            x,y=xy(pos);h=f['heading'][i];half=np.deg2rad(config.sensor_fov_deg/2);r=config.sensor_range/10*scale
            arc=[(x,y)]+[(x+r*np.cos(t),y-r*np.sin(t)) for t in np.linspace(h-half,h+half,16)]+[(x,y)]
            d.line(arc,fill='#4b7d81',width=1)
        for report in data.get('initial_intel', {}).get('objects', []):
            x,y=xy(np.asarray(report['position'])/10)
            for width,color in [(7,'#101826'),(3,'#e1bcff')]:
                d.line((x-11,y,x+11,y),fill=color,width=width)
                d.line((x,y-11,x,y+11),fill=color,width=width)
            d.ellipse((x-2,y-2,x+2,y+2),fill='#f7eaff')
        tracks={tuple(np.round(t['position'],4)):t for t in f['tracks'] if t['work_units']>0}
        occupied=[]
        upper_type2=max((t['position'][1] for t in f['truth'] if t['type']==2 and t['work_units']>0),default=None)
        for t in f['truth']:
            if t['work_units']<=0:continue
            x,y=xy(t['position']);detected=tuple(np.round(t['position'],4)) in tracks
            color='#ffd16e' if detected else '#e49bbd';r=5
            if detected:d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='#fff0c6',width=1)
            else:
                # A hollow diamond distinguishes ground truth from a detection.
                d.line([(x,y-r-1),(x+r+1,y),(x,y+r+1),(x-r-1,y),(x,y-r-1)],fill=color,width=2)
            label=f'Type {t["type"]}';ft=font(14);w=d.textlength(label,font=ft)
            offset=30 if t['type']==2 and t['position'][1]==upper_type2 else 12
            options=[(x+offset,y-20),(x+offset,y+8),(x-w-12,y-20),(x-w-12,y+8)]
            options += [(x+12,y+8+22*k) for k in range(1,8)]
            chosen=None
            for lx,ly in options:
                lx=float(np.clip(lx,6,size-w-7));ly=float(np.clip(ly,6,size-25))
                rect=(lx-4,ly-2,lx+w+4,ly+19)
                if not any(rect[0]<v[2] and rect[2]>v[0] and rect[1]<v[3] and rect[3]>v[1] for v in occupied):
                    chosen=(lx,ly,rect);break
            if chosen is None:chosen=(lx,ly,rect)
            lx,ly,rect=chosen;occupied.append(rect)
            if abs(ly-y)>28:d.line((x,y,lx,ly+8),fill='#8293a8',width=1)
            d.rounded_rectangle(rect,radius=3,fill='#132033')
            d.text((lx,ly),label,font=ft,fill=color)
        for i,pos in enumerate(f['pos']):
            if not f['active'][i]:continue
            x,y=xy(pos);h=f['heading'][i]
            # Directional drone glyphs, without drone text labels.
            v=np.array([np.cos(h),-np.sin(h)]);n=np.array([-v[1],v[0]])
            tip=np.array([x,y])+v*7;back=np.array([x,y])-v*5
            d.polygon([tuple(tip),tuple(back+n*4),tuple(back-n*4)],fill=COLORS[i%len(COLORS)])
        d.rectangle((0,0,size-1,size-1),outline='#43566f',width=1)
        im.paste(layer,(ox+30,90))
        draw.text((ox+30,765),'Belief',font=font(14),fill=MUTED)
        for k in range(100):
            c=tuple((low+(high-low)*k/99).astype(int));draw.line((ox+82+k,769,ox+82+k,779),fill=c)
        draw.text((ox+190,765),'Low → High',font=font(14),fill=MUTED)
        draw.text((ox+345,765),'◇ Actual position',font=font(14),fill='#e49bbd')
        draw.text((ox+500,765),'● Detected',font=font(14),fill='#ffd16e')
        draw.text((ox+30,800),'+ Initial report',font=font(14),fill='#c89bff')
        draw.text((ox+190,800),'Polygons: candidates / assignments',font=font(14),fill=MUTED)
    return im

def video(a,b,path,config):
    process=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1440x850','-r','12','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for index in range(max(len(a['frames']),len(b['frames']))):
            im=render_frame(a,b,index,config)
            process.stdin.write(im.tobytes())
            if index==min(40,max(len(a['frames']),len(b['frames']))-1):
                im.save(Path(path).with_suffix('.png'))
    finally:process.stdin.close()
    if process.wait():raise RuntimeError('ffmpeg failed')
