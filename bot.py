import asyncio
import os
import string
from datetime import datetime
from typing import List

import discord
import replicate
import replicate.exceptions
from discord import Intents, app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

model_dict = {
    "stable diffusion": "stability-ai/stable-diffusion",
    "pokemon": "lambdal/text-to-pokemon",
    "pixel art": "andreasjansson/monkey-island-sd",
    "logo": "laion-ai/erlich",
    "anime": "cjwbw/waifu-diffusion",
    # "dalle mini": "kuprel/min-dalle",
}

guild_id = os.environ["GUILD_ID"]
intents = Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    description="Runs models on Replicate!",
    intents=intents,
)

tree = bot.tree


async def model_autocomplete(
    ctx: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=model, value=model)
        for model in model_dict
        if model.lower().startswith(current)
    ]


@tree.command(name="test", guild=discord.Object(id=guild_id))
@discord.ui.button(label="test", style=discord.ButtonStyle.blurple)
async def _test(ctx: discord.Interaction):
    await ctx.response.send_message("howdy!")
    await ctx.edit_original_response(view=discord.ui.Button())
    # await ctx.edit_original_response(content="howdy!")


async def _predict(model, **kwargs):
    return model.predict(**kwargs)


@tree.command(
    name="flip",
    description="Dream up an image with a specified model",
    guild=discord.Object(id=guild_id),
)
@app_commands.autocomplete(model=model_autocomplete)
async def dream(ctx: discord.Interaction, model: str, prompt: str):
    """Generate an image from a text prompt using the stable-diffusion model"""
    model_string = model_dict[model]
    _model = replicate.models.get(model_string)
    await ctx.response.send_message(f"“{prompt}”\n> Generating ({model})...")

    # Request
    image = None
    try:
        if model == "logo":
            image = list(await _predict(_model, prompt=prompt, batch_size=1))[0][0]
        elif model == "dalle mini":
            image = list(
                await _predict(
                    _model, prompt=prompt, progressive_outputs=False, grid_size=2
                )
            )
        else:
            image = list(await _predict(_model, prompt=prompt))[0]
    except replicate.exceptions.ModelError as e:
        await ctx.edit_original_response(
            content=f"“{prompt}” ({model}) **failed**. This is most likely due to the model generating an NSFW image. Try again."
        )
        print(e)
        return
    except Exception as e:
        await ctx.edit_original_response(
            content=f"“{prompt}” ({model}) **failed** for some reason. This is probably my fault."
        )
        print(e)
        return

    # Response
    if image:
        author = ctx.user
        embed = (
            discord.Embed(
                title=f"“{prompt}”",
                # description=f"Generated with {model}",
                color=discord.Color.random(),
            )
            .set_author(
                name=author.display_name,
                icon_url=author.avatar.url,
            )
            .set_footer(
                text=f"Image generated using {model} model.",
                icon_url=bot.user.avatar.url,
            )
        )
        embed.set_image(url=image)
        await ctx.edit_original_response(
            # content=f"“{prompt}” ({model})\n{image}"
            content=None,
            embed=embed,
        )
    else:
        await ctx.edit_original_response(content=f"“{prompt}” ({model}) **failed**.")


@bot.event
async def on_ready():
    tree.clear_commands(guild=None)
    await tree.sync(guild=discord.Object(id=guild_id))
    print("ready!")


bot.run(os.environ["DISCORD_TOKEN"])
