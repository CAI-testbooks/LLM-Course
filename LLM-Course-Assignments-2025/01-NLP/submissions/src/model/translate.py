import random
import hashlib
import urllib.parse
import http.client
import json


class BaiduGeneralTranslator:

    def __init__(self):
        self.appid = '20251117002499767'  # 保持你的appid
        self.secret_key = '_CTthjZOiAoDdFlg3gTJ'  # 保持你的密钥
        self.api_url = "/api/trans/vip/translate"  # 通用翻译API端点
        # 移除domain参数（通用翻译不需要）

    def translate(self, text):
        # 非中文文本直接返回
        if not any('\u4e00' <= c <= '\u9fff' for c in text):
            return text

        # 准备API参数（通用翻译无需domain）
        from_lang = "zh"
        to_lang = "en"
        salt = random.randint(32768, 65536)
        # 签名字符串不再包含domain
        sign_str = f"{self.appid}{text}{salt}{self.secret_key}"

        # 计算签名
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

        # 构造请求参数（移除domain）
        params = {
            "appid": self.appid,
            "q": text,
            "from": from_lang,
            "to": to_lang,
            "salt": salt,
            "sign": sign
        }
        url_params = urllib.parse.urlencode(params)
        full_url = f"{self.api_url}?{url_params}"

        # 发送HTTP请求（通用翻译仍使用api.fanyi.baidu.com域名）
        http_client = None
        try:
            http_client = http.client.HTTPConnection("api.fanyi.baidu.com")
            http_client.request("GET", full_url)

            # 解析响应（通用翻译的返回格式与领域翻译一致）
            response = http_client.getresponse()
            result = response.read().decode('utf-8')
            result_json = json.loads(result)

            if "trans_result" in result_json:
                translated_text = result_json["trans_result"][0]["dst"]
                print(f"中文→英文：{text} → {translated_text}")
                return translated_text
            else:
                error_msg = result_json.get("error_msg", "未知错误")
                print(f"失败：{error_msg}（错误码：{result_json.get('error_code')}）")
                return text

        except Exception as e:
            print(f"请求异常：{str(e)}")
            return text
        finally:
            if http_client:
                http_client.close()