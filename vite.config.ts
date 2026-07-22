import { defineConfig } from "vite";
const coi={"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
export default defineConfig({base:process.env.BASE_PATH||"/",worker:{format:"es"},server:{headers:coi},preview:{headers:coi}});
