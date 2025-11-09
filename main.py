from astrbot.api.message_components import Plain, Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import aiohttp
import asyncio
import random
import json


@register("ms_aiimg", "", "接入魔搭社区文生图模型。支持LLM调用和命令调用。", "1.2")
class ModFlux(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.size = config.get("size", "1080x1920")
        self.api_url = config.get("api_url")
        self.provider = config.get("provider", "ms")  # 默认为ModelScope

        if not self.api_key:
            raise ValueError("API密钥必须配置")

    async def _request_modelscope(self, prompt: str, size: str, session: aiohttp.ClientSession) -> str:
        """向ModelScope API发送请求"""
        common_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        current_seed = random.randint(1, 2147483647)
        payload = {
            "model": f"{self.model}",
            "prompt": prompt,
            "seed": current_seed,
            "size": size,
            "num_inference_steps": "30",
        }
        
        async with session.post(
            f"{self.api_url}v1/images/generations",
            headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        ) as response:
            response.raise_for_status()
            task_response = await response.json()
            task_id = task_response.get("task_id")
            
            if not task_id:
                raise Exception("未能获取任务ID，生成图片失败。")

        # 使用指数退避策略轮询结果
        delay = 1
        max_delay = 10
        while True:
            async with session.get(
                f"{self.api_url}v1/tasks/{task_id}",
                headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
            ) as result_response:
                result_response.raise_for_status()
                data = await result_response.json()

                task_status = data.get("task_status")
                if task_status == "SUCCEED":
                    output_images = data.get("output_images", [])
                    if output_images:
                        return output_images[0]
                    else:
                        raise Exception("图片生成成功但未返回图片URL。")
                elif task_status == "FAILED":
                    raise Exception("图片生成失败。")
                
                # 指数退避策略
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def _request_image(self, prompt: str, size: str) -> str:
        """
        根据配置的提供商向不同的API发起请求，返回图片URL。
        """
        try:
            if not prompt:
                raise ValueError("请提供提示词！")

            async with aiohttp.ClientSession() as session:
                if self.provider.lower() == "ms" or self.provider.lower() == "modelscope":
                    return await self._request_modelscope(prompt, size, session)
                else:
                    raise ValueError(f"不支持的提供商: {self.provider}")

        except aiohttp.ClientError as e:
            raise Exception(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"解析API响应失败: {str(e)}")
        except Exception as e:
            raise e

    @filter.llm_tool(name="draw")
    async def draw(self, event: AstrMessageEvent, prompt: str, size: str = "1080x1920"):
        '''根据提示词生成图片，需要包含主体,场景,风格等必要提示词

        Args:
            prompt(string): 图片提示词
            size(string): 图片尺寸，如1920x1080
        '''
        
        try:
            # 根据提供商发送不同的请求
            image_url = await self._request_image(prompt, size)

            # 添加一个奇怪的检查，不知道为什么加了这行代码就正常了，或许是赛博佛祖罢
            if self is None:
                yield event.plain_result("你还不够虔诚，所以没有得到佛祖的庇佑导致发生了错误")
                return
            
            # 拿到结果后，再发送最终的图片消息
            chain = [Image.fromURL(image_url)]
            yield event.chain_result(chain)

        except Exception as e:
            yield event.plain_result(f"生成图片时遇到问题: {str(e)}")
            
    @filter.command("aiimg")
    async def generate_image_command(self, event: AstrMessageEvent):
        # 获取用户输入的提示词
        full_message = event.message_obj.message_str
        parts = full_message.split(" ", 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""

        if not prompt:
            yield event.plain_result("\n请提供提示词！使用方法：/aiimg <提示词>")
            return

        try:
            current_seed = random.randint(1, 2147483647)

            # 根据提供商发送不同的请求
            image_url = await self._request_image(prompt, self.size)
            
            # 拿到结果后，再发送最终的图片消息
            chain = [
                Plain(f"提示词：{prompt}\n"),
                Image.fromURL(image_url)
            ]
            yield event.chain_result(chain)

        except Exception as e:
            yield event.plain_result(f"\n生成图片失败: {str(e)}")
