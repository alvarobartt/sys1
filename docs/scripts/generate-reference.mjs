import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const docsDir = fileURLToPath(new URL('../', import.meta.url));
const manifest = fileURLToPath(new URL('../../Cargo.toml', import.meta.url));
const publicDir = path.join(docsDir, 'public');
const generatedDir = path.join(docsDir, '.vitepress', 'generated');
const notesDir = path.join(docsDir, '.vitepress', 'reference-notes');
mkdirSync(generatedDir, { recursive: true });

const manifestText = readFileSync(manifest, 'utf8');
const packageSection = manifestText.split(/^\[package\]\s*$/m)[1]?.split(/^\[/m)[0];
const crateVersion = packageSection?.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
if (!crateVersion) throw new Error('Could not read [package].version from Cargo.toml');

const openapiPath = path.join(publicDir, 'openapi.json');
const spec = JSON.parse(readFileSync(openapiPath, 'utf8'));
if (spec.info?.version !== crateVersion) {
  throw new Error(
    `docs/public/openapi.json has info.version ${JSON.stringify(spec.info?.version)}, ` +
    `but Cargo.toml has version ${crateVersion}. Upload the OpenAPI JSON for this crate version.`
  );
}

function cargo(args) {
  const result = spawnSync('cargo', ['run', '--quiet', '--locked', '--manifest-path', manifest, ...args], {
    cwd: docsDir,
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`cargo run ${args.join(' ')} failed:\n${result.stderr.trim()}`);
  }
  return result.stdout;
}

const help = cargo(['--', '--help']);
writeFileSync(path.join(publicDir, 'cli-help.txt'), help);

function escapeCell(value) {
  return String(value).replaceAll('|', '\\|').replaceAll('\n', ' ');
}

function optionGroups(helpText) {
  const groups = new Map();
  let group = null;
  let option = null;
  for (const line of helpText.split(/\r?\n/)) {
    const heading = line.match(/^([A-Z][A-Za-z ]+):$/);
    if (heading) {
      group = heading[1];
      option = null;
      groups.set(group, []);
      continue;
    }
    const flag = line.match(/^ {2,6}(?:(-\w), )?(--[a-z0-9-]+)(?: <[^>]+>)?(?: {2,}(\S.*))?$/);
    if (flag && group) {
      option = { flag: [flag[1], flag[2]].filter(Boolean).join(', '), help: flag[3] || '' };
      groups.get(group).push(option);
      continue;
    }
    const detail = line.match(/^\s{10}(\S.*)$/);
    if (detail && option) option.help += `${option.help ? ' ' : ''}${detail[1]}`;
  }
  const count = [...groups.values()].reduce((sum, options) => sum + options.length, 0);
  if (count < 10 || ![...groups.values()].flat().some(({ flag }) => flag.includes('--batch-wait-ms'))) {
    throw new Error('Could not parse the expected sys1 options from Clap help');
  }
  return groups;
}

const cliHeadings = {
  Options: 'General',
  'Model options': 'Model',
  'Server options': 'Server',
  'Batching options': 'Batching and limits'
};
const cliLines = ['<!-- Generated from sys1 --help. Edit Rust help or docs/cli.md, not this file. -->', ''];
for (const [heading, options] of optionGroups(help)) {
  cliLines.push(`## ${cliHeadings[heading] || heading}`, '', '| Flag | Environment | Default | Description |', '| --- | --- | --- | --- |');
  for (const option of options) {
    const env = option.help.match(/\[env: ([A-Z0-9_]+)=\]/)?.[1] || 'None';
    const fallback = option.help.match(/\[default: ([^\]]+)\]/)?.[1] || 'None';
    const values = option.help.match(/\[possible values: ([^\]]+)\]/)?.[1];
    const description = option.help.replace(/\s*\[(?:env|default|possible values): [^\]]+\]/g, '').trim();
    const detail = values ? `${description.replace(/\.?$/, '.')} Values: ${values}.` : description;
    cliLines.push(`| \`${option.flag}\` | \`${env}\` | \`${fallback}\` | ${escapeCell(detail)} |`);
  }
  cliLines.push('');
}
writeFileSync(path.join(generatedDir, 'cli.md'), `${cliLines.join('\n')}\n`);

const operations = new Map();
for (const [route, methods] of Object.entries(spec.paths)) {
  for (const [method, operation] of Object.entries(methods)) {
    operations.set(`${method.toUpperCase()} ${route}`, operation);
  }
}
const primary = [
  ['Health', 'GET /health', 'health'],
  ['List models', 'GET /v1/models', 'models'],
  ['System One', 'POST /v1/systemone', 'system-one'],
  ['Metrics', 'GET /metrics', 'metrics']
];
const aliases = new Map([['POST /v1/systemone', ['POST /v1/decide']]]);
const known = new Set(primary.map(([, route]) => route));
for (const aliasRoutes of aliases.values()) for (const route of aliasRoutes) known.add(route);
for (const [route, operation] of operations) {
  if (!known.has(route)) primary.push([route, route, null]);
}

function typeName(schema = {}) {
  if (schema.$ref) return schema.$ref.split('/').at(-1);
  if (Array.isArray(schema.type)) return schema.type.map((type) => typeName({ type })).join(' or ');
  if (schema.type === 'array') return `array of ${typeName(schema.items)}`;
  if (schema.type === 'object') return 'object';
  return schema.type || 'any JSON';
}

function fields(schema) {
  if (!schema) return [];
  const resolved = schema.$ref ? spec.components.schemas[schema.$ref.split('/').at(-1)] : schema;
  if (!resolved?.properties) return [];
  const required = new Set(resolved.required || []);
  return Object.entries(resolved.properties).map(([name, property]) => [name, typeName(property), required.has(name) ? 'Yes' : 'No']);
}

function schemaFromContent(content) {
  return Object.values(content || {})[0]?.schema;
}

const apiLines = ['<!-- Generated from docs/public/openapi.json. Edit that file or reference-notes, not this file. -->', ''];
for (const [title, route, note] of primary) {
  const operation = operations.get(route);
  if (!operation) throw new Error(`OpenAPI is missing ${route}`);
  apiLines.push(`## ${title}`, '', `\`${route}\``);
  const aliasRoutes = aliases.get(route) || [];
  for (const aliasRoute of aliasRoutes) {
    const alias = operations.get(aliasRoute);
    if (!alias) throw new Error(`OpenAPI is missing ${aliasRoute}`);
    if (JSON.stringify(alias.requestBody) !== JSON.stringify(operation.requestBody)
      || JSON.stringify(alias.responses) !== JSON.stringify(operation.responses)) {
      throw new Error(`${aliasRoute} no longer matches ${route}`);
    }
    apiLines.push('', `Alias: \`${aliasRoute}\`.`);
  }
  if (operation.summary) apiLines.push('', operation.summary);

  const body = operation.requestBody;
  if (body) {
    const content = Object.keys(body.content || {}).join(', ') || 'None';
    apiLines.push('', `Request content type: \`${content}\`.`, '');
    const requestFields = fields(schemaFromContent(body.content));
    if (requestFields.length) {
      apiLines.push('### Request fields', '', '| Field | Type | Required |', '| --- | --- | --- |');
      for (const [name, type, required] of requestFields) {
        apiLines.push(`| \`${name}\` | ${escapeCell(type)} | ${required} |`);
      }
      apiLines.push('');
    }
  }

  apiLines.push('', '### Responses', '', '| Status | Content type | Meaning |', '| --- | --- | --- |');
  for (const [status, response] of Object.entries(operation.responses || {})) {
    const content = Object.keys(response.content || {}).join(', ') || 'None';
    apiLines.push(`| ${status} | \`${content}\` | ${escapeCell(response.description || '')} |`);
  }
  apiLines.push('');
  const success = operation.responses?.['200'];
  const responseFields = fields(schemaFromContent(success?.content));
  if (responseFields.length) {
    apiLines.push('### Response fields', '', '| Field | Type |', '| --- | --- |');
    for (const [name, type] of responseFields) apiLines.push(`| \`${name}\` | ${escapeCell(type)} |`);
    apiLines.push('');
  }
  if (note) {
    const notesPath = path.join(notesDir, `${note}.md`);
    if (existsSync(notesPath)) apiLines.push(readFileSync(notesPath, 'utf8').trim(), '');
  }
}
writeFileSync(path.join(generatedDir, 'api.md'), `${apiLines.join('\n')}\n`);
