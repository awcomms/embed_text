import { pipeline } from "@xenova/transformers";
import { koaBody } from "koa-body";
import { Buffer } from "node:buffer";
import koa from "koa";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/bert-base-uncased",
  { revision: "default" }
);

new koa()
  .use(koaBody())
  .use(async (ctx) => {
    try {
      const f = (
      await extractor(await ctx.request.body, {
        pooling: "mean",
        normalize: true,
      })
    ).data;
    ctx.body = ctx.request.headers["b"] ? Buffer.from(f.buffer) : Array.from(f);
    } catch (e) {
      console.error('k error', e)
      ctx.status = 500
      ctx.body = ''
    }
  })
  .listen(
    10000,
    process.env.NODE_ENV === "production" ? "0.0.0.0" : "127.0.0.1"
  );
