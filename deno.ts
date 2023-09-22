import { pipeline } from "npm:@xenova/transformers@2.6.1";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/bert-base-uncased",
  { revision: "default" }
);

const handler = async (request: Request): Promise<Response> => {
  if (
    request.method === "POST" &&
    request.headers.get("content-type") === "application/json"
  ) {
    const text = request.json();

    const output = await extractor(text, { pooling: "mean", normalize: true });
    return new Response(JSON.stringify(output), {
      status: 200,
      headers: {
        "content-type": "application/json",
      },
    });
  } else {
    return new Response(null, {
      status: request.method !== "POST" ? 405 : 415,
      headers: {
        "content-type": "application/text",
      },
    });
  }
};

Deno.serve({
  hostname:
    Deno.env.get("ENVIRONMENT") === "development" ? "127.0.0.1" : "0.0.0.0",
  port: 10000,
  handler,
});
