from astrbot.api.message_components import *
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import aiohttp
import asyncio
import random
import json

@register("ms_aiimg", "竹和木", "接入魔搭社区文生图模型。使用 /aiimg <提示词> 生成图片。", "1.0")
class ModFlux(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.num_inference_steps = config.get("num_inference_steps")
        self.size = config.get("size")
        self.api_url = config.get("api_url")
        self.seed = config.get("seed")
        self.enable_translation = config.get("enable_translation")  # 保留此配置项以维持兼容性

        if not self.api_key:
            raise ValueError("API密钥必须配置")

    @filter.command("aiimg")
    async def generate_image(self, event: AstrMessageEvent):
        # 获取用户输入的提示词
        full_message = event.message_obj.message_str
        parts = full_message.split(" ", 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""

        if not self.api_key:
            yield event.plain_result("\n请先在配置文件中设置API密钥")
            return

        if not prompt:
            yield event.plain_result("\n请提供提示词！使用方法：/aimg <提示词>")
            return

        try:
            try:
                if self.seed == "随机" or not self.seed:
                    current_seed = random.randint(1, 2147483647)
                else:
                    current_seed = int(self.seed)
            except (ValueError, TypeError):
                current_seed = random.randint(1, 2147483647)

            # 调用ModelScope API
            base_url = 'https://api-inference.modelscope.cn/'
            api_key = self.api_key  # 使用配置中的API密钥

            common_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # 发起异步请求
            async with aiohttp.ClientSession() as session:
                # 提交图片生成任务
                async with session.post(
                    f"{base_url}v1/images/generations",
                    headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                    data=json.dumps({
                        "model": f"{self.model}",
                        "prompt": prompt
                    }, ensure_ascii=False).encode('utf-8')
                ) as response:
                    response.raise_for_status()
                    task_response = await response.json()
                    task_id = task_response["task_id"]

                while True:
                    async with session.get(
                        f"{base_url}v1/tasks/{task_id}",
                        headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                    ) as result_response:
                        result_response.raise_for_status()
                        data = await result_response.json()

                        if data["task_status"] == "SUCCEED":
                            image_url = data["output_images"][0]
                            chain = [
                                Plain(f"提示词：{prompt}\nseed ID：{current_seed}\n生成完成"),
                                Image.fromURL(image_url)
                            ]
                            yield event.chain_result(chain)
                            break
                        elif data["task_status"] == "FAILED":
                            yield event.plain_result("\n图片生成失败。")
                            break

                        await asyncio.sleep(5)

        except Exception as e:
            yield event.plain_result(f"\n生成图片失败: {str(e)}")
