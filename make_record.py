"""Run on Windows with reportlab and Pillow after both browser tests."""
import json
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from PIL import Image
from reportlab.lib.utils import ImageReader

read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
result=read('browser-results.json')
live=read('live-results.json')
meta=read('site/fixtures/manifest.json')
pdfmetrics.registerFont(TTFont('cn','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('bd','C:/Windows/Fonts/msyhbd.ttc'))
c=Canvas('test-record.pdf',pagesize=(595,842))
c.setTitle('mini-ddpm-web 浏览器部署测试记录')
def line(s,x,y,size=10,bold=False):
    c.setFillColor(HexColor('#20383a'));c.setFont('bd' if bold else 'cn',size);c.drawString(x,y,s)
def para(s,y):
    p=Paragraph(s,ParagraphStyle('cntext',fontName='cn',fontSize=10.3,leading=17,wordWrap='CJK'))
    _,h=p.wrap(505,800);p.drawOn(c,45,y-h)
def page(title,n):
    line('MINI-DDPM-WEB    /    2026.11.20',45,795,10,True)
    line(title,45,750,22,True)
    c.line(45,728,550,728);line('浏览器推理测试记录',45,25,8);line(f'{n} / 2',515,25,8)
def table(rows,widths,y):
    for i,row in enumerate(rows):
        if i%2==0:
            c.setFillColor(HexColor('#e5f1ed'));c.rect(45,y-24,sum(widths),24,fill=1,stroke=0)
        x=49
        for val,w in zip(row,widths):
            line(str(val),x,y-16,9,i==0);x+=w
        y-=24
page('从训练权重到浏览器生成',1)
para('设备为 Windows 笔记本、NVIDIA GeForce RTX 4060 Laptop GPU，驱动 591.86。实测 Chrome '+result['browser_version']+'，WebGPU 已启用，浏览器返回 NVIDIA / Lovelace 适配器。U-Net 使用 ONNX Runtime Web 1.23.2 的 WebGPU 后端，部分形状运算可由 CPU 执行。',706)
line('两份模型的接口',45,612,13,True)
para('两份 ONNX 均从前两题的 EMA 权重导出。输入 x_t 与输出 epsilon 均为 float32 [1,1,28,28]。时间 t 为 int64 [1]。条件模型多接收一个 int64 [1] 的 label，取值 0–9，整条轨迹保持不变。',588)
line('单步噪声预测误差',45,505,13,True)
para('固定初始噪声文件、t=999、条件标签 7。先将 Python ONNX Runtime CPU 与 PyTorch CPU 比较，再将浏览器输出与同一 PyTorch 参考比较。下表统计 784 个位置的绝对差。',485)
rows=[['实现','模式','平均绝对误差','最大绝对误差']]
modes=[('unconditional','无条件'),('conditional','指定数字')]
for platform in ['Python ONNX','Chrome WebGPU']:
    for key,label in modes:
        v=meta[key] if platform=='Python ONNX' else result['modes'][key]['eps']
        rows.append([platform,label,f"{v['mean']:.3e}",f"{v['max']:.3e}"])
table(rows,[150,105,125,120],419)
line('发布页面实测',45,261,13,True)
para('通过 HTTPS 打开 banligegit.github.io/mini-ddpm/，HTTP 200。新浏览器会话依次生成指定数字 7 和随机类别，均使用网页种子 2026、20 步。下图取自实际页面 Canvas。',239)
for i,(key,label) in enumerate(reversed(modes)):
    x=60+260*i
    im=Image.open('live-'+key+'.png').resize((140,140),Image.Resampling.NEAREST)
    c.drawImage(ImageReader(im),x,104,70,70)
    line(label,x+79,137,10,True)
    line(live['measurements'][key],x,78,8.5)
para('首次加载含模型获取与建会话。先加载的模式还承担运行时初始化，公网速度受网络和缓存影响。',56)
c.showPage()
page('跨端对齐与种子复测',2)
para('Python 保存一份 float32 小端原始噪声，两端读取完全相同的 3,136 字节。两端核对每一步时间编号，沿用 T=1000 线性 beta、[-1,1] 截断及截断后重算噪声方向。η=0，最后从编号 0 到干净端点。',706)
line('完整 DDIM 轨迹',45,626,13,True)
para('参考值来自 PyTorch CPU。浏览器通过本机静态服务加载同一页面，完成一次单步预测后逐项测量。耗时为每张图的采样循环时间。像素差按 Canvas 的 0–255 灰度量化逐点比较。',604)
rows=[['模式','步数','张量平均差','张量最大差','不同像素 / 784','采样 / ms']]
for key,label in modes:
    for k,v in result['modes'][key]['steps'].items():
        rows.append([label,k,f"{v['tensor']['mean']:.3e}",f"{v['tensor']['max']:.3e}",v['changedPixels'],f"{v['ms']:.1f}"])
table(rows,[76,55,101,101,100,67],525)
line('网页种子与异常提示',45,267,13,True)
para('固定条件 7、10 步，网页种子 2026 连续生成两次，最终张量逐元素相同。改为 2027 后，平均绝对差为 %.5f。这个测试采用网页的高斯随机数生成器；上表的跨语言实验直接读取噪声文件。'%result['otherSeed']['mean'],245)
para('在独立页面模拟 navigator.gpu 缺失，页面显示检查浏览器支持及硬件加速的提示，生成按钮保持禁用。实测环境支持 WebGPU，真实硬件不支持时的表现仅覆盖上述模拟分支。',168)
para('重现命令、导出依赖及 Pages 配置见 README.md。完整结果见 browser-results.json、live-results.json 和 site/fixtures/manifest.json。以上误差针对本次固定输入。',94)
c.save()
