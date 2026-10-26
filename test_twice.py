import time
import app
print("llm:", app.llm)
t=time.time()
r1=app.ask_cebri("test-user", "hello")
print("reply1:", r1)
print("time1:", round(time.time()-t,2))
t=time.time()
r2=app.ask_cebri("test-user", "who are you?")
print("reply2:", r2)
print("time2:", round(time.time()-t,2))
