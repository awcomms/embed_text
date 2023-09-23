import { pipeline } from "@xenova/transformers";
import { koaBody } from "koa-body";
import {Buffer } from 'node:buffer'
import koa from "koa";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/bert-base-uncased",
  { revision: "default" }
);

new koa()
  .use(koaBody())
  .use(async (ctx) => {
    const text = await ctx.request.body;
    ctx.body = Buffer.from((
      await extractor(text, { pooling: "mean", normalize: true })
    ).data);
  })
  .listen(
    10000,
    process.env.NODE_ENV === "production" ? "0.0.0.0" : "127.0.0.1"
  );
