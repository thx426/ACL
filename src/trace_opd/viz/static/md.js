// Tiny markdown renderer for research doc.md files (headings, lists, bold,
// italics, inline code, fenced code). Input is escaped first — docs are data.

const esc = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\*([^*]+)\*/g, "<i>$1</i>");
}

export function renderMarkdown(text) {
  const out = [];
  let list = false, code = false, codeBuf = [];
  const closeList = () => { if (list) { out.push("</ul>"); list = false; } };
  for (const raw of text.split("\n")) {
    if (raw.startsWith("```")) {
      if (code) { out.push(`<pre>${esc(codeBuf.join("\n"))}</pre>`); codeBuf = []; }
      code = !code;
      continue;
    }
    if (code) { codeBuf.push(raw); continue; }
    const line = raw.trimEnd();
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(li[1])}</li>`); continue; }
    closeList();
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  if (code && codeBuf.length) out.push(`<pre>${esc(codeBuf.join("\n"))}</pre>`);
  return out.join("\n");
}
