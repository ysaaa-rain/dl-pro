"""Run local-only annotation UI: .venv/bin/python scripts/annotation_server.py."""
import argparse,json,secrets,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from dlpro.annotation_tool import AnnotationEngine,ReviewStore
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--database',type=Path,default=ROOT/'data/annotations/human_reviews.sqlite3');args=parser.parse_args();engine=AnnotationEngine(ROOT);store=ReviewStore(args.database);predictions={};lock=threading.Lock();token=secrets.token_urlsafe(24)
 class Handler(BaseHTTPRequestHandler):
  def send(self,status,data,kind='application/json; charset=utf-8'):
   body=data.encode() if isinstance(data,str) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
  def do_GET(self):
   if self.headers.get('Host') not in [f'127.0.0.1:{args.port}',f'localhost:{args.port}']:return self.send(403,dict(error='仅允许本机访问'))
   path=urlparse(self.path).path
   if path=='/':return self.send(200,(ROOT/'src/dlpro/annotation.html').read_text().replace('__TOKEN__',token),'text/html; charset=utf-8')
   if path=='/api/reviews':return self.send(200,store.records())
   if path=='/api/export':return self.send(200,store.export_csv(),'text/csv; charset=utf-8')
   return self.send(404,dict(error='路径不存在'))
  def do_POST(self):
   if self.headers.get('Host') not in [f'127.0.0.1:{args.port}',f'localhost:{args.port}']:return self.send(403,dict(error='仅允许本机访问'))
   if self.headers.get('X-Session-Token')!=token:return self.send(403,dict(error='无效本地会话'))
   try:
    length=int(self.headers.get('Content-Length','0'))
    if not 0<length<=20000:raise ValueError('请求大小无效')
    body=json.loads(self.rfile.read(length))
    if not isinstance(body,dict):raise ValueError('请求必须是JSON对象')
    path=urlparse(self.path).path
    with lock:
     if path=='/api/predict':
      prediction=engine.predict(body.get('center'));key=secrets.token_urlsafe(18);predictions[key]=prediction
      if len(predictions)>8:predictions.pop(next(iter(predictions)))
      return self.send(200,dict(**prediction,prediction_id=key))
     if path=='/api/review':
      key=body.get('prediction_id')
      if key not in predictions:raise ValueError('预测已失效，请重新预测后保存')
      identity=store.save(predictions[key],body.get('decision'),body.get('note',''));return self.send(200,dict(id=identity))
     return self.send(404,dict(error='路径不存在'))
   except (ValueError,TypeError,KeyError) as e:return self.send(400,dict(error=str(e)))
   except FileNotFoundError:return self.send(503,dict(error='该坐标需要原始WT矩阵；请运行数据恢复命令或使用已缓存示例坐标'))
   except Exception as e:
    print(repr(e),flush=True);return self.send(500,dict(error='推理失败，请查看服务器日志'))
 print(f'Annotation UI: http://127.0.0.1:{args.port}',flush=True)
 try:ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
 except KeyboardInterrupt:print('Annotation server stopped.',flush=True)
if __name__=='__main__':main()
