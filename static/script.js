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
  const data = await response.json();
  // Update this line to match the key 'processed_query' used in your JSON response
  document.getElementById("answerSpace").innerText = data.processed_query;
}
