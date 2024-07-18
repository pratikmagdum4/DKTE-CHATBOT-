class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatbox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            voiceButton: document.querySelector('.chatbox__voice-button'),
            voiceImg: document.getElementById('voice-img')
        };
        this.state = false;
        this.messages = [];
        this.isListening = false;
    }

    display() {
        const { openButton, chatbox, sendButton, voiceButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatbox));
        sendButton.addEventListener('click', () => this.onSendButton(chatbox));
        const node = chatbox.querySelector('input');
        // voiceButton.addEventListener('click', () => this.toggleVoiceInput());
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatbox);
            }
        });
    }

    toggleVoiceInput() {
        this.isListening = !this.isListening;

        if (this.isListening) {
            this.startListening();
        } else {
            this.stopListening();
        }
    }

    startListening() {
        console.log("startListening....")

        // Change microphone image to micro-on
        this.args.voiceImg.src = "{{ url_for('static', filename='images/micro-on.png') }}";

        this.sendMessageToServer("voice_input");
    }

    stopListening() {
        console.log("stopListening....")
        // Change microphone image to micro-off
        this.args.voiceImg.src = "{{ url_for('static', filename='images/micro-off.png') }}";
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    sendMessageToServer(message) {
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "Sam", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText();
            }).catch((error) => {
                console.error('Error:', error);
            });
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "Sam", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox);
                textField.value = '';
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
                textField.value = '';
            });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach((item, index) => {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();


