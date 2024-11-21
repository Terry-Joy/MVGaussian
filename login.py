from huggingface_hub import login

# 替换为您的Hugging Face API Token
api_token = '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'

def hugging_face_login(token):
    """
    使用给定的API Token登录Hugging Face。
    
    参数:
    - token (str): Hugging Face API Token
    
    返回:
    - None
    """
    try:
        login(token)
        print("成功登录到 Hugging Face!")
    except Exception as e:
        print(f"登录失败: {e}")

# 调用函数进行登录
hugging_face_login(api_token)