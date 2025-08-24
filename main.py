from astrbot.api.message_components import Plain, Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import aiohttp
import asyncio
import random
import json


@register("ms_aiimg", "", "接入魔搭社区文生图模型。使用 /aiimg <提示词> 生成图片。", "1.0")
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

        if not prompt:
            yield event.plain_result("\n请提供提示词！使用方法：/aiimg <提示词>")
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
            base_url = self.api_url
            api_key = self.api_key

            common_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # 发起异步请求
            async with aiohttp.ClientSession() as session:
                # 提交图片生成任务
                payload = {
                    "model": f"{self.model}",
                    "prompt": prompt,
                    "seed": current_seed
                }
                
                async with session.post(
                    f"{base_url}v1/images/generations",
                    headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
                ) as response:
                    response.raise_for_status()
                    task_response = await response.json()
                    task_id = task_response.get("task_id")
                    
                    if not task_id:
                        yield event.plain_result("\n未能获取任务ID，生成图片失败。")
                        return

                # 使用指数退避策略轮询结果
                delay = 1
                max_delay = 10
                while True:
                    async with session.get(
                        f"{base_url}v1/tasks/{task_id}",
                        headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                    ) as result_response:
                        result_response.raise_for_status()
                        data = await result_response.json()

                        task_status = data.get("task_status")
                        if task_status == "SUCCEED":
                            output_images = data.get("output_images", [])
                            if output_images:
                                image_url = output_images[0]
                                chain = [
                                    Plain(f"提示词：{prompt}\nseed ID：{current_seed}\n生成完成"),
                                    Image.fromURL(image_url)
                                ]
                                yield event.chain_result(chain)
                                break
                            else:
                                yield event.plain_result("\n图片生成成功但未返回图片URL。")
                                break
                        elif task_status == "FAILED":
                            yield event.plain_result("\n图片生成失败。")
                            break
                        
                        # 指数退避策略
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, max_delay)

        except aiohttp.ClientError as e:
            yield event.plain_result(f"\n网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            yield event.plain_result(f"\n解析API响应失败: {str(e)}")
        except Exception as e:
            yield event.plain_result(f"\n生成图片失败: {str(e)}")
