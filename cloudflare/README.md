# Serving naija-petro.shinzii.tech via Cloudflare

Modal custom domains need a paid plan, so the app is served on the custom domain
with a free Cloudflare Worker that reverse-proxies to the Modal deployment
(`https://peniel-tish--naija-petro.modal.run`). The `.modal.run` URL keeps
working too.

## One-time setup

1. **Add the domain to Cloudflare.** Create a free Cloudflare account, "Add a
   site" `shinzii.tech`, pick the Free plan. Cloudflare imports the existing DNS
   and shows you two nameservers.
2. **Switch nameservers at Namecheap.** Domain List > Manage `shinzii.tech` >
   Nameservers > **Custom DNS**, enter Cloudflare's two nameservers, save. Wait
   for Cloudflare to mark the zone **Active** (minutes to a few hours).
3. **Create the Worker.** Cloudflare dashboard > Workers & Pages > Create >
   Worker. Replace its code with `worker.js` here and Deploy.
4. **Attach the custom domain.** In the Worker > Settings > Domains & Routes >
   Add > **Custom domain** > `naija-petro.shinzii.tech`. Cloudflare creates the
   DNS record and TLS certificate automatically.

After the certificate is issued (a minute or two), `https://naija-petro.shinzii.tech`
serves the app, including streaming and the admin panel at `/admin`.

## Notes

- If the Modal deployment URL ever changes, update `UPSTREAM` in `worker.js` and
  redeploy the Worker.
- The Worker forwards `CF-Connecting-IP` as `X-Forwarded-For` so the app's
  per-IP daily free limit still counts real visitors.
