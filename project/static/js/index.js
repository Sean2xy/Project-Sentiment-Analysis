

document.addEventListener("DOMContentLoaded", () => {
  const inputField = document.getElementById("input");
  inputField.addEventListener("keydown", (e) => {
    if (e.code === "Enter") {
      let input = inputField.value;
      inputField.value = "";
      output(input);
    }
  });
});

function output(input) {
  let product;

  // Regex remove non word/space chars
  // Trim trailing whitespce
  // Remove digits - not sure if this is best
  // But solves problem of entering something like 'hi1'

  let text = input.trim();
  console.log(text)

  var xhr = new XMLHttpRequest();

  xhr.open("POST", "http://127.0.0.1:8000", true);
  
  xhr.setRequestHeader('Content-Type', 'application/json');
  //xhr.setRequestHeader('Content-length', '8' );

  xhr.onreadystatechange = function() {//Call a function when the state changes.

    if(xhr.readyState == 4 && xhr.status == 200) {
    console.log(xhr.responseText)
    //alert(xhr.responseText);
    addChat(input, xhr.responseText);
    }
    
    }
  xhr.send(JSON.stringify({
    user_input: text
  }));

  
}


function addChat(input, product) {
  const messagesContainer = document.getElementById("messages");

  let userDiv = document.createElement("div");
  userDiv.id = "user";
  userDiv.className = "user response";
  userDiv.innerHTML = `<img src="../static/images/user.png" class="avatar"><span>${input}</span>`;
  messagesContainer.appendChild(userDiv);

  let botDiv = document.createElement("div");
  let botImg = document.createElement("img");
  let botText = document.createElement("span");
  botDiv.id = "bot";
  botImg.src = "../static/images/robot.png";
  botImg.className = "avatar";
  botDiv.className = "bot response";
  botText.innerText = "Typing...";
  botDiv.appendChild(botText);
  botDiv.appendChild(botImg);
  messagesContainer.appendChild(botDiv);
  // Keep messages at most recent
  messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;

  // Fake delay to seem "real"
  setTimeout(() => {
    botText.innerText = `${product}`;
    textToSpeech(product)
  }, 2000
  )

}