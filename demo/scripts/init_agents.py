import http.client
import json

# List of agent ports to register
agent_ports = ["localhost:8000", "localhost:8001", "localhost:8004", "localhost:10000"]

conn = http.client.HTTPConnection("localhost", 12000)
headers = {
  'Content-Type': 'application/json'
}

# Register each agent
for agent_port in agent_ports:
    payload = json.dumps({
        "params": agent_port
    })
    
    conn.request("POST", "/agent/register", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(f"Registered {agent_port}: {data.decode('utf-8')}")

conn.close()