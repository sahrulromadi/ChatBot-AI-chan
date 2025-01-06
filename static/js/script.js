function sendMessage() {
    var userMessage = document.getElementById("userInput").value;
    if (userMessage.trim() === "") return;

    appendMessage(userMessage, "user");

    // AJAX
    fetch(`/chatbot/?msg=${userMessage}`)
        .then((response) => response.json())
        .then((data) => {
            var botResponse = data.response;
            appendMessage(botResponse, "bot");
        })
        .catch((error) => console.log("Error:", error));

    // clear input
    document.getElementById("userInput").value = "";
}

function appendMessage(message, sender) {
    var chatBox = document.getElementById("chatBox");
    var messageElement = document.createElement("div");
    messageElement.classList.add("message");
    if (sender === "user") {
        messageElement.classList.add("user-message");
    } else {
        messageElement.classList.add("bot-message");
    }
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Keyboard Enter
document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});