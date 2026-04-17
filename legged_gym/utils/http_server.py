from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from threading import Thread
import json
import time
import numpy as np
import math

class MyHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate = True):

        self.vforward = 0
        self.vleftright = 0
        self.vrotate = 0

        #command
        self.joint_position_command=list(np.zeros(18))
        #observation
        self.imu_rpy_angle=list(np.zeros(3))
        self.imu_rotate_velocity=list(np.zeros(3))
        self.robot_angular_velocity=list(np.zeros(3))
        self.robot_projected_gravity=list(np.zeros(3))
        self.encoder_joint_position=list(np.zeros(18))
        self.encoder_joint_velocity=list(np.zeros(18))
        self.net_actions=list(np.zeros(18))
        self.joystick_command=list(np.zeros(3))
        self.joint_torque=list(np.zeros(18))
        self.contact_forces_z=list(np.zeros(6))

        super().__init__(server_address, RequestHandlerClass, bind_and_activate)


class MyRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.server:MyHTTPServer=server
        super().__init__(request, client_address, server)

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        # Parse the query parameters from the URL
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)

        response_dict = {}
        if 'variable_name' in query_params:
            for param_name in query_params['variable_name']:
                if hasattr(self.server, param_name):
                    response_dict[param_name] = getattr(self.server, param_name)

        response_json = json.dumps(response_dict)
        self.wfile.write(response_json.encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # 解析 POST 数据
        post_str = post_data.decode('utf-8')
        post_dict = urllib.parse.parse_qs(post_str)
        
        updated=False
        message=""
        for var_name, var_value in post_dict.items():
            if hasattr(self.server, var_name):
                updated=True
                if len(var_value)==1:
                    setattr(self.server, var_name, float(var_value[0]))
                else:
                    setattr(self.server, var_name, var_value)
                message=message+"{} updated value: {}".format(var_name,var_value[0])+"\r\n"

        if updated:
            self._set_headers()
            self.wfile.write(message.encode())
        else:
            # 如果 POST 数据中缺少必要变量，则返回错误状态码
            self._set_headers(400)
            self.wfile.write("Required variables not found in POST data".encode())
            
    def log_message(self, format, *args):
        if (self.command == 'GET')or(self.command =="POST"):
            pass  # 不处理GET请求的日志消息
        else:
            super().log_message(format, *args)

if __name__ =="__main__":
    port=3861
    server_address = ('0.0.0.0', port)
    server = MyHTTPServer(server_address,
                          RequestHandlerClass=MyRequestHandler)
    
    print(f"Starting server on port {port}")
    thread1=Thread(target=server.serve_forever,daemon=True)
    thread1.start()

    count=0
    while True:
        # server.printSavedVariables()
        count=(count+1) % 100
        # server.joint_position_command=list(np.ones(18)*count/100)
        # server.encoder_joint_position=list(np.ones(18)*(100-count)/100)
        time.sleep(0.01)
    
    
