async function submitQueryOriginal() {
  const query = document.getElementById("queryInput").value;
  const response = await fetch("/query-original/", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `query=${encodeURIComponent(query)}`,
  });
  const data = await response.json();
  // Update this line to match the key 'processed_query' used in your JSON response
  document.getElementById("answerSpace").innerText = data.processed_query;
}

async function submitQueryRetrievalRerank() {
  const query = document.getElementById("queryInput").value;
  const response = await fetch("/query-retrieval-rerank/", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `query=${encodeURIComponent(query)}`,
  });
  const results = await response.json();
  const answerElement = document.getElementById("answer-text");
  answerElement.innerHTML = results.answer; // Assuming 'results.answer' contains the text of the answer
  const highlightedParagraph = results.best_paragraph.replace(
    results.answer,
    `<span class="highlight">${results.answer}</span>`
  );
  const sourceElement = document.getElementById("source-text");
  sourceElement.innerHTML = highlightedParagraph;
  document.getElementById("documentContent").innerHTML = results.best_document;
  let htmlContent = `<h3>Answer</h3><p>${results.answer} ${results.score}</p>`;
  htmlContent += `<h3>Source Paragraph</h3><p>${results.best_paragraph}</p>`;
  htmlContent += "<h3>Top-3 Cross-Encoder Re-ranker Hits</h3><ol>";

  results.cross_encoder_hits.forEach((hit) => {
    htmlContent += `<li>Score: ${hit.score.toFixed(3)} - Text: ${
      hit.text
    }</li>`;
  });

  htmlContent += "</ol><h3>Top-3 Bi-Encoder Retrieval Hits</h3><ol>";

  results.bi_encoder_hits.forEach((hit) => {
    htmlContent += `<li>Score: ${hit.score.toFixed(3)} - Text: ${
      hit.text
    }</li>`;
  });

  htmlContent += "</ol>";
  htmlContent += "<p>Te \n st</p>";
  document.getElementById("answerSpace").innerHTML = htmlContent;
  document.getElementById("toggleDocumentBtn").style.display = "block";
}

function toggleDocument() {
  var content = document.getElementById("documentContent");
  if (content.style.display === "none") {
    content.style.display = "block";
    document.getElementById("toggleDocumentBtn").innerText = "Hide Document";
  } else {
    content.style.display = "none";
    document.getElementById("toggleDocumentBtn").innerText =
      "Read the whole document here";
  }
}

// Bind the toggleDocument function to the button click event
document.getElementById("toggleDocumentBtn").onclick = toggleDocument;
