import {describe,expect,it} from 'vitest';import {normal,rgbaFromNchw} from './common';
describe('runtime utilities',()=>{it('generates deterministic noise',()=>expect([...normal(8,42)]).toEqual([...normal(8,42)]));it('converts NCHW RGB',()=>expect([...rgbaFromNchw(new Float32Array([1,0,0]),1,1)]).toEqual([255,128,128,255]))});
