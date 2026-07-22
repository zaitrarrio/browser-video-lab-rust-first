import {readFile} from 'node:fs/promises';
for(const file of process.argv.slice(2)){const m=JSON.parse(await readFile(file,'utf8'));for(const key of ['kind','width','height','models'])if(!(key in m))throw new Error(`${file}: missing ${key}`);console.log(`${file}: ${Object.keys(m.models).join(', ')}`)}
