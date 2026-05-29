// Cloudflare Worker: serve the Naija-Petro app on a custom domain
// (naija-petro.shinzii.tech) by reverse-proxying to the Modal deployment.
//
// Why a proxy: Modal routes by its own *.modal.run hostname and only holds a
// TLS certificate for that, so a plain CNAME cannot serve the app under another
// domain. This Worker terminates TLS for the custom domain at Cloudflare's edge
// and forwards each request to Modal with the correct upstream Host, streaming
// the response back unchanged (so Server-Sent Events keep working).
//
// Setup: deploy this Worker, then attach the custom domain
// naija-petro.shinzii.tech to it (Worker > Settings > Domains & Routes >
// Add > Custom domain). See cloudflare/README.md.

const UPSTREAM = "https://peniel-tish--naija-petro.modal.run";

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Copy incoming headers; drop Host so fetch() sets it (and the TLS SNI) to
    // the Modal upstream, which is how Modal routes to the app.
    const headers = new Headers(request.headers);
    headers.delete("host");

    // Preserve the real visitor IP so the app's per-IP daily limit still works.
    const clientIp = request.headers.get("CF-Connecting-IP");
    if (clientIp) headers.set("X-Forwarded-For", clientIp);

    const init = { method: request.method, headers, redirect: "manual" };
    // Buffer request bodies (small JSON, or uploads up to 8 MB); responses stay
    // streamed below, which is what matters for SSE.
    if (request.method !== "GET" && request.method !== "HEAD") {
      init.body = await request.arrayBuffer();
    }

    const resp = await fetch(UPSTREAM + url.pathname + url.search, init);

    // Stream the response straight through (keeps text/event-stream live).
    const respHeaders = new Headers(resp.headers);
    respHeaders.delete("content-length");
    return new Response(resp.body, {
      status: resp.status,
      statusText: resp.statusText,
      headers: respHeaders,
    });
  },
};
