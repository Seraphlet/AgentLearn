import json
from openai import OpenAI
import os
import urllib.request
def get_weather(city:str)->str: 
    city_encoded = urllib.parse.quote(city)          # 深圳 → %E6%B7%B1%E5%9C%B3
    url = f"https://wttr.in/{city_encoded}?format=3"
  # 返回简短的天气文本
    return urllib.request.urlopen(url).read().decode()

# 注册工具
TOOLS_SEARCH=[
     {
            "type":"function","function":{
                "name":"get_weather","description":"获取指定城市天气","parameters":{
                    "type":"object",
                    "properties":{
                        "city":{
                            "type":"string"
                        }
                    },
                     "required":["city"]
                        }
                }
            }  
]

def fc_loop(user_input,max_runs=3):
    messages=[{"role":"user","content":user_input}]
    client=OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")
    for _ in range(max_runs):
        res=client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools= TOOLS_SEARCH,
        max_tokens=100,
        )
        msg=res.choices[0].message
        try:
            print("msg:type",type(msg),"\n\n",msg)
        except:
            print("无法打印")
        if not msg.tool_calls:
            return msg.content
        #Sprint(msg)
        messages.append(msg)
        for tc in msg.tool_calls:
            #result=get_weather(**json.loads(tc.function.arguments))
            #print(tc)
            try:
                arg=json.loads(tc.function.arguments)
            except Exception as e:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"参数解析失败:{e}"})
                continue
            result=run_status(tc.function.name,arg)
            messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
            try:
                print("arg:", arg, "result:", result)
            except Exception as e:
                print("result无法打印",e)
    return "达到最大轮数,已停止"
TOOLS_FUNC = {"get_weather": get_weather}
def run_status(name:str,args:dict)->str:
    #执行工具;出错不崩,把错误信息返回给模型,让它自己修正
    try:
        if name not in TOOLS_FUNC:
            return f"未知工具:{name}"
        return TOOLS_FUNC[name](**args)
    except Exception as e:
        return f"工具执行失败:{e}"


if __name__=="__main__":
    print(fc_loop(input("天气助手:")))

