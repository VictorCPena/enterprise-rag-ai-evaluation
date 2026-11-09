'use strict';
const byId = (id) => document.getElementById(id);

function card(parent, title, text) {
  const article = document.createElement('article');
  const heading = document.createElement('h3');
  const body = document.createElement('pre');
  heading.textContent = title;
  body.textContent = text;
  article.append(heading, body);
  byId(parent).append(article);
  return article;
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
  return result;
}

byId('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  byId('answer').replaceChildren();
  byId('results').replaceChildren();
  byId('submit').disabled = true;
  const payload = {
    text: byId('q').value,
    k: 5,
    strategy: byId('strategy').value,
    mode: byId('mode').value,
  };
  try {
    const started = performance.now();
    const [answer, search] = await Promise.all([
      post('/query', payload), post('/search', payload),
    ]);
    byId('time').textContent = `${(performance.now() - started).toFixed(1)} ms browser round trip · ${answer.mode}`;
    card('answer', answer.abstained ? 'Insufficient evidence' : 'Evidence answer', answer.answer);
    for (const citation of answer.citations) {
      const article = card('answer', `Citation ${citation.id}`,
        `${citation.quote}\nSHA256 ${citation.source_sha256}\nCharacters ${citation.start}–${citation.end}`);
      const link = document.createElement('a');
      link.textContent = citation.doc;
      const url = new URL(citation.url);
      if (url.protocol === 'https:') link.href = url.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      article.append(link);
    }
    for (const result of search.results) {
      card('results', `${result.doc} · score ${result.score.toFixed(5)}`, result.text);
    }
  } catch (error) {
    card('answer', 'Request failed', error.message);
  } finally {
    byId('submit').disabled = false;
  }
});

byId('eval').addEventListener('click', async () => {
  try {
    const response = await fetch('/evaluation');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    byId('metrics').textContent = JSON.stringify(await response.json(), null, 2);
  } catch (error) {
    byId('metrics').textContent = error.message;
  }
});
