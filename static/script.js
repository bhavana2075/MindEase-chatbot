async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  const chatBox = document.getElementById("chat-box");
  addMessage("user", `🧍‍♂️: ${message}`);
  input.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, user_id: "default" }),
    });
    const data = await res.json();
    addMessage("bot", `🤖: ${data.reply}`);
  } catch {
    addMessage("bot", "🤖: Sorry, I'm having trouble responding right now.");
  }
}

function addMessage(role, text) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", role);
  msgDiv.innerHTML = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight; // ✅ keeps scroll at bottom
}
