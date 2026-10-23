const loading_document = document.querySelector(".loading");
const content = document.querySelector(".content");
const input = document.querySelector("input");
const button = document.querySelector(".send");

let messages = JSON.parse(localStorage.getItem("messages")) || [];

for (let el of messages) {
  content.innerHTML += `<p class="question">${el.question}</p>`;
  content.innerHTML += `<p class="answer">${el.answer}</p>`;
}

const getData = async () => {
  console.log("Clickd");
  const url = "http://localhost:8000";
  try {
      loading_document.classList.remove("none");
      let message = input.value;
      input.value = "";
      const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message, k: 5 }),
      });
      if (!response.ok) {
          console.log(await response.json());
          throw new Error(`Response Status: ${response.status}`);
      }
      const result = await response.json();
      let messages = JSON.parse(localStorage.getItem("messages")) || [];
      let new_msg = [...messages, { question: message, answer: result.message }];
      new_msg = new_msg.filter((el) => el.answer);
      localStorage.setItem("messages", JSON.stringify(new_msg));
      loading_document.classList.add("none");
      content.innerHTML = "";
      for (let el of new_msg) {
          content.innerHTML += `<p class="question">${el.question}</p>`;
          content.innerHTML += `<p class="answer">${el.answer}</p>`;
      }
  } catch (error) {
      console.error(error);
  }
};

function handler() {
    let messages = JSON.parse(localStorage.getItem("messages")) || [];
    let message = input.value;
    let new_msg = [...messages, { question: message, answer: "" }];
    localStorage.setItem("messages", JSON.stringify(new_msg));
    content.innerHTML = "";
    for (let el of new_msg) {
        content.innerHTML += `<p class="question">${el.question}</p>`;
        if (el.answer) content.innerHTML += `<p class="answer">${el.answer}</p>`;
    }
    getData();
}

function handlerEnter(e) {
    if (e.key == "Enter") {
        handler()
    }
}

button.addEventListener("click", handler);
document.addEventListener("keydown", handlerEnter);
