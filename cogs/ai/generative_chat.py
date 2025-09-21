from core.ai.core import ModelsList
from core.exceptions import *
from core.services.helperfunctions import HelperFunctions
from core.ai.history import History as typehint_History
from discord import Message
from os import environ
import aimodels._template_ as typehint_AIModelTemplate
import discord
import importlib
import inspect
import logging
import re


class BaseChat():
    def __init__(self, bot, author, history: typehint_History):
        self.bot: discord.Bot = bot
        self.author = author
        self.DBConn = history

        # Control de concurrencia
        self.pending_ids = []

    ###############################################
    # Events-based chat
    ###############################################
    async def _ask(self, prompt: Message):
        # Check if SHARED_CHAT_HISTORY is enabled
        if environ.get("SHARED_CHAT_HISTORY", "false").lower() == "true":
            guild_id = prompt.guild.id if prompt.guild else prompt.author.id
        else:
            guild_id = prompt.author.id

        # Set default model
        _model = await self.DBConn.get_default_model(guild_id=guild_id)
        if _model is None:
            logging.info("No default model found, using default model")
            _model = await self.DBConn.get_default_model(guild_id=guild_id)

        _model_provider = _model.split("::")[0]
        _model_name = _model.split("::")[-1]

        # Selección de modelo explícito
        if "/model:" in prompt.content:
            _modelUsed = await prompt.channel.send("🔍 Using specific model")
            async for _model_selection in ModelsList.get_models_list_async():
                _prov = _model_selection.split("::")[0]
                _name = _model_selection.split("::")[-1]

                if re.search(rf"/model:{_name}(\s|$)", prompt.content):
                    _model_provider, _model_name = _prov, _name
                    await _modelUsed.edit(content=f"🔍 Using model: **{_model_name}**")
                    break
            else:
                await _modelUsed.edit(content=f"🔍 Using model: **{_model_name}**")

        # Flags
        _append_history = True
        if "/chat:ephemeral" in prompt.content:
            await prompt.channel.send("🔒 This conversation is not saved and Jakey won't remember this")
            _append_history = False

        _show_info = "/chat:info" in prompt.content

        # Carga dinámica del modelo
        try:
            _infer: typehint_AIModelTemplate.Completions = importlib.import_module(
                f"aimodels.{_model_provider}"
            ).Completions(
                model_name=_model_name,
                discord_ctx=prompt,
                discord_bot=self.bot,
                guild_id=guild_id
            )
        except ModuleNotFoundError:
            raise CustomErrorMessage("⚠️ The model you've chosen is not available at the moment, please choose another model")

        _infer._discord_method_send = prompt.channel.send

        ###############################################
        # File attachment processing
        ###############################################
        if len(prompt.attachments) > 1:
            await prompt.reply("🚫 I can only process one file at a time")
            return

        if prompt.attachments:
            if not hasattr(_infer, "input_files"):
                raise CustomErrorMessage(f"🚫 The model **{_model_name}** cannot process file attachments, please try another model")

            _processFileInterstitial = await prompt.channel.send(f"📄 Processing the file: **{prompt.attachments[0].filename}**")
            await _infer.input_files(attachment=prompt.attachments[0])
            await _processFileInterstitial.edit(f"✅ Used: **{prompt.attachments[0].filename}**")

        ###############################################
        # Answer generation
        ###############################################
        _final_prompt = re.sub(
            rf"(<@{self.bot.user.id}>(\s|$)|/model:{_model_name}(\s|$)|/chat:ephemeral(\s|$)|/chat:info(\s|$))",
            "",
            prompt.content
        ).strip()

        _system_prompt = await HelperFunctions.set_assistant_type("jakey_system_prompt", type=0)

        async with prompt.channel.typing():
            _result = await _infer.chat_completion(
                prompt=_final_prompt,
                db_conn=self.DBConn,
                system_instruction=_system_prompt
            )

        if _result["response"] == "OK" and _show_info:
            await prompt.channel.send(
                embed=discord.Embed(
                    description=f"Answered by **{_model_name}** by **{_model_provider}** (this response isn't safe)"
                )
            )

        # Save to chat history
        if _append_history:
            if not hasattr(_infer, "save_to_history"):
                await prompt.channel.send("⚠️ This model doesn't allow saving the conversation")
            else:
                await _infer.save_to_history(db_conn=self.DBConn, chat_thread=_result["chat_thread"])

    ###############################################
    # Discord on_message event
    ###############################################
    async def on_message(self, message: Message):
        # Ignore messages from the bot itself
        if message.author.id == self.bot.user.id:
            return

        # Must be mentioned or DM
        if message.guild is None or self.bot.user.mentioned_in(message):
            # Ensure it must not be triggered by command prefix or slash command
            if message.content.startswith(self.bot.command_prefix) or message.content.startswith("/"):
                if message.content.startswith(self.bot.command_prefix):
                    _command = message.content.split(" ")[0].replace(self.bot.command_prefix, "")
                    if self.bot.get_command(_command):
                        return

            # Check if the user is in the pending list
            if message.author.id in self.pending_ids:
                await message.reply("⚠️ I'm still processing your previous request, please wait for a moment...")
                return

            # If only mention without content or attachments
            if not message.attachments and not re.sub(f"<@{self.bot.user.id}>", '', message.content).strip():
                return

            # Remove the mention
            message.content = re.sub(f"<@{self.bot.user.id}>", '', message.content).strip()

            # Handle attachments
            if message.attachments:
                _alttext = message.attachments[0].description if message.attachments[0].description else "No alt text provided"
                message.content = inspect.cleandoc(f"""<extra_metadata>
                    <attachment url="{message.attachments[0].url}" />
                    <alt>
                        {_alttext}
                    </alt>
                </extra_metadata>

                {message.content}""")

            # Handle reply context
            if message.reference:
                _context_message = await message.channel.fetch_message(message.reference.message_id)
                message.content = inspect.cleandoc(
                    f"""<reply_metadata>
                    
                    # Replying to referenced message excerpt from {_context_message.author.display_name} (username: @{_context_message.author.name}):
                    <|begin_msg_contexts|diff>
                    {_context_message.content}
                    <|end_msg_contexts|diff>
                    
                    <constraints>Do not echo this metadata, only use for retrieval purposes</constraints>
                    </reply_metadata>
                    {message.content}"""
                )
                await message.channel.send(f"✅ Referenced message: {_context_message.jump_url}")

            try:
                # Add the user to the pending list
                self.pending_ids.append(message.author.id)

                # Add reaction
                await message.add_reaction("⌛")
                await self._ask(message)

            except Exception as _error:
                if isinstance(_error, HistoryDatabaseError):
                    await message.reply(f"🤚 Database error: **{_error.message}**")
                elif isinstance(_error, ModelAPIKeyUnset):
                    await message.reply(f"⛔ Model unavailable: **{_error.message}**")
                elif isinstance(_error, discord.errors.HTTPException) and "Cannot send an empty message" in str(_error):
                    await message.reply("⚠️ I received an empty response, please rephrase your question or try another model")
                elif isinstance(_error, CustomErrorMessage):
                    await message.reply(f"{_error.message}")
                else:
                    await message.reply(f"🚫 Sorry, I couldn't answer right now. Error: **{type(_error).__name__}**")

                logging.error("An error has occurred while generating an answer", exc_info=True)

            finally:
                # Remove the reaction
                try:
                    await message.remove_reaction("⌛", self.bot.user)
                except Exception:
                    pass

                # Remove the user from the pending list safely
                if message.author.id in self.pending_ids:
                    self.pending_ids.remove(message.author.id)
