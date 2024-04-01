"""
This script adds endpoint to the ComfyUI server
"""

import os

import folder_paths


def initiarize():
    import server

    web = server.web

    @server.PromptServer.instance.routes.post("/upload/audio")
    async def upload_audio(request):
        pass

    @server.PromptServer.instance.routes.get("/playaudio")
    async def play_audio(request):
        query = request.rel_url.query
        if "filename" not in query:
            return web.Response(status=404)
        filename = query["filename"]

        filename, output_dir = folder_paths.annotated_filepath(filename)

        type = request.rel_url.query.get("type", "output")
        if type == "path":
            # special case for path_based nodes
            # NOTE: output_dir may be empty, but non-None
            output_dir, filename = os.path.split(filename)
        if output_dir is None:
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return web.Response(status=400)

        if "subfolder" in request.rel_url.query:
            output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)
        with open(file, "rb") as f:
            data = f.read()

        return web.Response(body=data, content_type="audio/wav")
