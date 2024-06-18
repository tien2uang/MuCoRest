import time

class Counter:
    def __init__(self):
        self.status_total = 0
        self.status_2xx = 0
        self.status_4xx = 0
        self.status_500 = 0

    def request(self, flow):
        with open("ncs.txt", "a") as f:
            f.write("========REQUEST========\n")
            f.write(flow.request.method + "\n")
            f.write(flow.request.pretty_url + "\n")
            f.write(flow.request.text + "\n")
    def response(self, flow):
        with open("ncs.txt", "a") as f:
            f.write("========RESPONSE========\n")
            f.write(str(time.time()) + "\n")
            f.write(str(flow.response.status_code) + "\n")
            f.write(flow.response.text + "\n")


addons = [Counter()]
