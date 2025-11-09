# # Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License.

# import sys
# import traceback
# from datetime import datetime
# from http import HTTPStatus

# from aiohttp import web
# from aiohttp.web import Request, Response, json_response
# from botbuilder.core import (
#     TurnContext,
# )
# from botbuilder.core.integration import aiohttp_error_middleware
# from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
# from botbuilder.schema import Activity, ActivityTypes

# from bots import EchoBot
# from config import DefaultConfig

# CONFIG = DefaultConfig()

# # Create adapter.
# # See https://aka.ms/about-bot-adapter to learn more about how bots work.
# ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))


# # Catch-all for errors.
# async def on_error(context: TurnContext, error: Exception):
#     # This check writes out errors to console log .vs. app insights.
#     # NOTE: In production environment, you should consider logging this to Azure
#     #       application insights.
#     print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
#     traceback.print_exc()

#     # Send a message to the user
#     await context.send_activity("The bot encountered an error or bug.")
#     await context.send_activity(
#         "To continue to run this bot, please fix the bot source code."
#     )
#     # Send a trace activity if we're talking to the Bot Framework Emulator
#     if context.activity.channel_id == "emulator":
#         # Create a trace activity that contains the error object
#         trace_activity = Activity(
#             label="TurnError",
#             name="on_turn_error Trace",
#             timestamp=datetime.utcnow(),
#             type=ActivityTypes.trace,
#             value=f"{error}",
#             value_type="https://www.botframework.com/schemas/error",
#         )
#         # Send a trace activity, which will be displayed in Bot Framework Emulator
#         await context.send_activity(trace_activity)


# ADAPTER.on_turn_error = on_error

# # Create the Bot
# BOT = EchoBot()


# # Listen for incoming requests on /api/messages
# async def messages(req: Request) -> Response:
#     return await ADAPTER.process(req, BOT)


# APP = web.Application(middlewares=[aiohttp_error_middleware])
# APP.router.add_post("/api/messages", messages)

# if __name__ == "__main__":
#     try:
#         web.run_app(APP, host="localhost", port=CONFIG.PORT)
#     except Exception as error:
#         raise error



import sys
import traceback
from datetime import datetime
from http import HTTPStatus
 
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    TurnContext,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes
 
# from bots import EchoBot
from bots import ThinkpalmRAGBot
# from bots import ThinkpalmCosmosRAGGraph
from config import DefaultConfig
 
CONFIG = DefaultConfig()
 
# Create adapter.
# See https://aka.ms/about-bot-adapter to learn more about how bots work.
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))
 
 
# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
    # This check writes out errors to console log .vs. app insights.
    # NOTE: In production environment, you should consider logging this to Azure
    #       application insights.
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
 
    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
        # Create a trace activity that contains the error object
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        # Send a trace activity, which will be displayed in Bot Framework Emulator
        await context.send_activity(trace_activity)
 
 
ADAPTER.on_turn_error = on_error
 
# Create the Bot
BOT = ThinkpalmRAGBot()
 
 
# Listen for incoming requests on /api/messages
async def messages(req: Request) -> Response:
    return await ADAPTER.process(req, BOT)
 
 
APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

# Debug endpoint to view retrieved documents
async def debug_retrieval(req: Request) -> Response:
    """Debug endpoint to view recent retrieval logs and documents."""
    try:
        import os
        from pathlib import Path
        
        # Read the verification file
        debug_file = Path("Verification_retrieved_docs.txt")
        if not debug_file.exists():
            return json_response({
                "error": "No debug file found. Make sure the bot has processed at least one message.",
                "file_path": str(debug_file.absolute())
            })
        
        # Read last N entries (last 50KB or last 5 entries)
        content = debug_file.read_text(encoding="utf-8")
        
        # Split by separator and get last few entries
        entries = content.split("==============================")
        recent_entries = entries[-6:] if len(entries) > 6 else entries  # Last 5 entries
        
        # Parse each entry
        parsed_entries = []
        for entry in recent_entries:
            if not entry.strip():
                continue
            entry_data = {
                "raw": entry.strip(),
                "timestamp": None,
                "question": None,
                "docs_count": None,
                "answer": None
            }
            # Extract key info
            for line in entry.split("\n"):
                if "Timestamp:" in line:
                    entry_data["timestamp"] = line.split("Timestamp:")[-1].strip()
                elif "Question:" in line:
                    entry_data["question"] = line.split("Question:")[-1].strip()
                elif "Retrieved Docs:" in line:
                    try:
                        entry_data["docs_count"] = int(line.split("Retrieved Docs:")[-1].strip())
                    except:
                        pass
                elif "Answer:" in line and not entry_data["answer"]:
                    # Get answer (might be multi-line)
                    answer_lines = entry.split("Answer:")[-1].strip()
                    entry_data["answer"] = answer_lines.split("==============================")[0].strip()
            
            parsed_entries.append(entry_data)
        
        return json_response({
            "status": "success",
            "total_entries": len(entries) - 1,
            "recent_entries": parsed_entries[-5:],  # Last 5
            "file_path": str(debug_file.absolute()),
            "file_size": len(content)
        })
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

# Endpoint to view full debug file
async def debug_file_view(req: Request) -> Response:
    """View the full debug file content."""
    try:
        from pathlib import Path
        debug_file = Path("Verification_retrieved_docs.txt")
        if not debug_file.exists():
            return json_response({"error": "Debug file not found"}, status=404)
        
        content = debug_file.read_text(encoding="utf-8")
        # Return as plain text
        return Response(text=content, content_type="text/plain")
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

# Health check endpoint
async def health_check(req: Request) -> Response:
    """Health check endpoint."""
    return json_response({"status": "healthy", "service": "thinkpalm-bot"})

APP.router.add_get("/debug/retrieval", debug_retrieval)
APP.router.add_get("/debug/file", debug_file_view)
APP.router.add_get("/health", health_check)
 
if __name__ == "__main__":
    try:
        web.run_app(APP, host="0.0.0.0", port=CONFIG.PORT)
    except Exception as error:
        raise error