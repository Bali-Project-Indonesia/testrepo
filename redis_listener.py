import redis
import json
from index_builder import rebuild_faiss_index, delete_from_faiss

r = redis.Redis(host='localhost', port=6379, db=0)
pubsub = r.pubsub()
pubsub.subscribe("job_updates")

print("👂 Listening to Redis...")

for message in pubsub.listen():
    if message['type'] == 'message':
        payload = json.loads(message['data'])
        action = payload.get('action')
        job_id = payload.get('job_id')

        print(f"📡 Action received: {action} for Job ID {job_id}")

        if action in ['posted', 'updated']:
            print("🔁 Rebuilding FAISS index...")
            rebuild_faiss_index()
        elif action == 'deleted':
            print("🗑️ Removing job from FAISS index...")
            delete_from_faiss(job_id)