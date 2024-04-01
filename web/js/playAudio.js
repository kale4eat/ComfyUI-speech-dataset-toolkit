import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

app.registerExtension({
    name: "SpeechDatasetToolkit.playAudio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDT_PlayAudio") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                onNodeCreated?.apply(this, arguments);
                this.default_widgets_count = this.widgets?.length || 0;
                this.populateAudioWidgets = (audios) => {
                    // clear audio widgets
                    if (this.widgets) {
                        for (let i = 0; i < this.widgets.length; i++) {
                            if (this.widgets[i].name.startsWith("_audio")) {
                                this.widgets[i].onRemove?.();
                            }
                        }
                        this.widgets.length = this.default_widgets_count;
                    }

                    // add widgets
                    let index = 0
                    for (const audio of audios) {
                        for (const params of audio) {
                            // create audio elem for node
                            const audioElement = document.createElement("audio");
                            audioElement.controls = true;
                            audioElement.loop = false;
                            audioElement.muted = false;

                            const previewWidget = this.addDOMWidget("_audio" + index, "playaudio", audioElement, {
                                serialize: false,
                                hideOnZoom: false,
                            });
                            const sourceURL = api.apiURL('/playaudio?' + new URLSearchParams(params));
                            audioElement.src = sourceURL;
                            index++
                        }
                    };

                    requestAnimationFrame(() => {
                        const sz = this.computeSize();
                        if (sz[0] < this.size[0]) {
                            sz[0] = this.size[0];
                        }
                        if (sz[1] < this.size[1]) {
                            sz[1] = this.size[1];
                        }
                        this.onResize?.(sz);
                        app.graph.setDirtyCanvas(true, false);
                    });
                }
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.audios) {
                    this.populateAudioWidgets(message.audios);
                }
            };
        }
    },
});