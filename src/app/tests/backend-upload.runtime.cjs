const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const root = path.resolve(__dirname, '..');
function compile(source, filename) {
  const result = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
    },
    reportDiagnostics: true,
  });
  assert.equal((result.diagnostics || []).length, 0);
  return result.outputText;
}

const apiPath = path.join(root, 'src/api.ts');
const apiModule = { exports: {} };
const requests = [];
let response;
const fetchStub = async (url, options) => {
  requests.push({ url, options });
  return typeof response === 'function' ? response(url, options) : response;
};
new Function('exports', 'require', 'fetch', compile(fs.readFileSync(apiPath, 'utf8'), apiPath))(
  apiModule.exports,
  name => {
    assert.equal(name, './features/collections/collectionImageConfig');
    return { COLLECTION_IMAGE_SIZE: 512 };
  },
  fetchStub,
);
const api = apiModule.exports.default;
const apiAst = ts.createSourceFile(apiPath, fs.readFileSync(apiPath, 'utf8'), ts.ScriptTarget.Latest, true);
const uuidHelper = apiAst.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'createSecureClientSessionId');
assert.ok(uuidHelper);
const loadUuidHelper = crypto => new Function(
  'globalThis',
  'exports',
  `${compile(uuidHelper.getText(apiAst), apiPath)}\nreturn createSecureClientSessionId;`,
)({ crypto }, {});
let randomValueCalls = 0;
const fallbackUuid = loadUuidHelper({
  getRandomValues(bytes) {
    randomValueCalls++;
    return require('node:crypto').randomFillSync(bytes);
  },
});
const firstUuid = fallbackUuid();
assert.match(firstUuid, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
assert.notEqual(fallbackUuid(), firstUuid);
assert.equal(randomValueCalls, 2, 'HTTP contexts without randomUUID must use cryptographic getRandomValues');
assert.throws(loadUuidHelper(undefined), /Secure random number generation is unavailable/);
assert.throws(loadUuidHelper({}), /Secure random number generation is unavailable/);

const managerPath = path.join(root, 'src/components/BackendManager.tsx');
const manager = fs.readFileSync(managerPath, 'utf8');
const ast = ts.createSourceFile(managerPath, manager, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const helper = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'backendIsInUse');
assert.ok(helper);
const backendIsInUse = new Function(
  `${compile(helper.getText(ast), managerPath)}\nreturn backendIsInUse;`,
)();
assert.equal(backendIsInUse([], 'engine', 'cpu'), false);
assert.equal(backendIsInUse([{ recipe: 'other' }], 'engine', 'cpu'), false);
assert.equal(backendIsInUse([{ recipe: 'engine' }], 'engine', 'cpu'), true);
assert.equal(backendIsInUse([{ recipe: 'engine', recipe_options: { backend: 'cpu' } }], 'engine', 'cpu'), true);
assert.equal(backendIsInUse([{ recipe: 'engine', recipe_options: { engine_backend: 'gpu' } }], 'engine', 'cpu'), false);
assert.match(manager, /info\.upload_supported === true && \(/);
assert.match(manager, /info\.installation_source === 'manual'/);
assert.match(manager, /info\.action !== 'upload'/);
assert.match(manager, /disabled=\{installing !== null \|\| inUse \|\| showBackendProgress\}/);
assert.match(manager, /await api\.health\(\)[\s\S]*?backendIsInUse\(health\.all_models_loaded[\s\S]*?await api\.uploadBackend/);
assert.match(manager, /aria-describedby="backend-upload-warning"/);
assert.match(manager, /Trust and upload/);

function testRiskAcknowledgementDialog() {
  const declaration = ast.statements.find(node => ts.isVariableStatement(node)
    && node.declarationList.declarations.some(item => item.name.getText(ast) === 'BackendUploadDialog'));
  assert.ok(declaration);
  const states = [];
  let cursor = 0;
  let submitted;
  const jsx = (type, props) => ({ type, props });
  const dialog = new Function('require', 'exports', 'hooks', `
    const { useState, useRef, useEffect, useI18n, useFocusTrap, RECIPE_LABELS,
      WorkspaceActionGroup, WorkspaceActionButton, isBackendArchiveUrl } = hooks;
    ${compile(declaration.getText(ast), managerPath)}
    return BackendUploadDialog;
  `)(
    name => {
      assert.equal(name, 'react/jsx-runtime');
      return { jsx, jsxs: jsx, Fragment: 'fragment' };
    },
    {},
    {
      useState(initial) {
        const index = cursor++;
        if (!(index in states)) states[index] = initial;
        return [states[index], value => { states[index] = value; }];
      },
      useRef: () => ({ current: null }),
      useEffect() {},
      useI18n: () => ({ t: value => value }),
      useFocusTrap() {},
      RECIPE_LABELS: {},
      WorkspaceActionGroup: 'action-group',
      WorkspaceActionButton: 'action-button',
      isBackendArchiveUrl: apiModule.exports.isBackendArchiveUrl,
    },
  );
  const render = (recipe = 'strata', disabled = false) => {
    cursor = 0;
    const tree = dialog({
      target: { recipe, backend: 'rocm' }, disabled, busy: disabled,
      onClose() {}, onUpload: (...args) => { submitted = args; },
    });
    const nodes = [];
    const visit = node => {
      if (Array.isArray(node)) node.forEach(visit);
      else if (node?.props) { nodes.push(node); visit(node.props.children); }
    };
    visit(tree);
    return nodes;
  };
  let nodes = render();
  const checkbox = nodes.find(node => node.props.type === 'checkbox');
  assert.equal(checkbox.props.checked, false, 'Risk acceptance must never default to checked');
  checkbox.props.onChange({ target: { checked: true } });
  const archive = new File(['archive'], 'engine.tar.gz');
  nodes.find(node => node.props.type === 'file').props.onChange({ target: { files: [archive] } });
  nodes = render();
  const install = nodes.find(node => node.props.appearance === 'primary');
  assert.equal(install.props.disabled, false);
  install.props.onClick();
  assert.equal(submitted[0].archive, archive);
  assert.equal(submitted[1], true, 'Checkbox acknowledgement must reach the installation handler');
  assert.equal(render('strata', true).find(node => node.props.type === 'checkbox').props.disabled, true);
  states.length = 0;
  assert.equal(render().find(node => node.props.type === 'checkbox').props.checked, false,
    'New installations require fresh acknowledgement');
  assert.equal(render('llamacpp').some(node => node.props.type === 'checkbox'), false);
}

async function main() {
  testRiskAcknowledgementDialog();
  api.baseUrl = 'https://offline-server.example:8443';
  api.apiKey = 'regular-test-key';
  api.setSessionAdminApiKey('admin-test-key');
  const archive = new File(['archive bytes'], 'engine.tar.gz');
  await assert.rejects(api.uploadBackend('custom engine', 'cpu&gpu', archive, false), /Confirm/);
  await assert.rejects(api.uploadBackend('engine', 'cpu', new File(['bad'], 'engine.zip'), true), /\.tar\.gz/);
  await assert.rejects(api.uploadBackend('engine', 'cpu', new File([], 'engine.tar.gz'), true), /non-empty/);
  await assert.rejects(api.uploadBackend('engine', 'cpu', {
    name: 'engine.tar.gz',
    size: 4 * 1024 ** 3 + 1,
    slice() { assert.fail('Oversized archive must not be read'); },
  }, true), /4 GiB/);
  assert.equal(requests.length, 0);

  response = Response.json({ status: 'success', recipe: 'custom engine', backend: 'cpu&gpu' });
  await api.uploadBackend('custom engine', 'cpu&gpu', archive, true);
  const { url, options } = requests.at(-1);
  const parsedUrl = new URL(url);
  assert.equal(parsedUrl.origin + parsedUrl.pathname, 'https://offline-server.example:8443/internal/backends/upload');
  assert.equal(parsedUrl.searchParams.get('recipe'), 'custom engine');
  assert.equal(parsedUrl.searchParams.get('backend'), 'cpu&gpu');
  assert.match(parsedUrl.searchParams.get('upload_id'), /^[0-9a-f-]{36}$/);
  assert.equal(parsedUrl.searchParams.get('offset'), '0');
  assert.equal(parsedUrl.searchParams.get('total'), String(archive.size));
  assert.equal(parsedUrl.searchParams.has('allow_unverified_release'), false);
  assert.equal(options.method, 'POST');
  assert.ok(options.body instanceof Blob, 'Send raw slices, not buffered or serialized data');
  assert.equal(await options.body.text(), await archive.text());
  assert.equal(options.headers['Content-Type'], 'application/octet-stream');
  assert.equal(options.headers['X-Lemonade-Trust-Executable'], 'true');
  assert.equal(options.headers.Authorization, 'Bearer admin-test-key');

  api.setSessionAdminApiKey('');
  response = Response.json({ status: 'success', recipe: 'engine', backend: 'cpu' });
  await api.uploadBackend('engine', 'cpu', archive, true);
  assert.equal(requests.at(-1).options.headers.Authorization, 'Bearer regular-test-key');

  for (const status of [400, 403, 409, 413, 500]) {
    response = Response.json({ error: 'archive rejected by server' }, { status });
    await assert.rejects(api.uploadBackend('engine', 'cpu', archive, true), /archive rejected by server/);
  }
  response = Response.json({ error: 'validation failed' });
  await assert.rejects(api.uploadBackend('engine', 'cpu', archive, true), /validation failed/);
  response = Response.json({ status: 'success', recipe: 'wrong', backend: 'cpu' });
  await assert.rejects(api.uploadBackend('engine', 'cpu', archive, true), /Unexpected/);

  const chunkSize = 32 * 1024 ** 2;
  const slices = [];
  const largeArchive = {
    name: 'large.tar.gz',
    size: chunkSize * 2 + 7,
    slice(start, end) {
      const slice = new Blob([`${start}:${end}`]);
      slices.push({ start, end, slice });
      return slice;
    },
  };
  const startIndex = requests.length;
  const controller = new AbortController();
  let acknowledgeFirst;
  response = (url) => {
    const offset = Number(new URL(url).searchParams.get('offset'));
    if (offset === 0) return new Promise(resolve => { acknowledgeFirst = resolve; });
    return offset === chunkSize
      ? Response.json({ status: 'uploading', received: chunkSize * 2, total: largeArchive.size }, { status: 202 })
      : Response.json({ status: 'success', recipe: 'engine', backend: 'cpu' });
  };
  const upload = api.uploadBackend('engine', 'cpu', largeArchive, true, controller.signal);
  assert.equal(requests.length - startIndex, 1, 'Do not send the next chunk before acknowledgement');
  assert.equal(slices.length, 1, 'Slice only the current chunk');
  acknowledgeFirst(Response.json({ status: 'uploading', received: chunkSize, total: largeArchive.size }, { status: 202 }));
  await upload;
  const chunks = requests.slice(startIndex);
  assert.equal(chunks.length, 3);
  assert.deepEqual(slices.map(({ start, end }) => [start, end]), [
    [0, chunkSize], [chunkSize, chunkSize * 2], [chunkSize * 2, largeArchive.size],
  ]);
  const ids = new Set();
  chunks.forEach(({ url, options }, index) => {
    const params = new URL(url).searchParams;
    ids.add(params.get('upload_id'));
    assert.equal(params.get('offset'), String(slices[index].start));
    assert.equal(params.get('total'), String(largeArchive.size));
    assert.equal(options.body, slices[index].slice);
    assert.equal(options.signal, controller.signal);
    assert.equal(options.headers['X-Lemonade-Trust-Executable'], 'true');
    assert.equal(options.headers['Content-Type'], 'application/octet-stream');
    assert.equal(options.headers.Authorization, 'Bearer regular-test-key');
  });
  assert.equal(ids.size, 1, 'Reuse the upload ID across chunks');
  assert.notEqual([...ids][0], parsedUrl.searchParams.get('upload_id'));

  for (const acknowledgement of [
    { status: 'uploading', received: chunkSize - 1, total: largeArchive.size },
    { status: 'uploading', received: chunkSize, total: largeArchive.size + 1 },
    { status: 'success', recipe: 'engine', backend: 'cpu' },
  ]) {
    const before = requests.length;
    response = Response.json(acknowledgement, { status: 202 });
    await assert.rejects(api.uploadBackend('engine', 'cpu', largeArchive, true), /acknowledgement/);
    assert.equal(requests.length - before, 1, 'Malformed acknowledgement must stop the upload');
  }
  response = Response.json({ status: 'uploading', received: chunkSize, total: largeArchive.size });
  await assert.rejects(api.uploadBackend('engine', 'cpu', largeArchive, true), /acknowledgement/);
  response = Response.json({ status: 'success', recipe: 'engine', backend: 'cpu' }, { status: 202 });
  await assert.rejects(api.uploadBackend('engine', 'cpu', archive, true), /Unexpected/);

  let before = requests.length;
  response = url => Number(new URL(url).searchParams.get('offset')) === 0
    ? Response.json({ status: 'uploading', received: chunkSize, total: largeArchive.size }, { status: 202 })
    : Response.json({ error: 'chunk rejected' }, { status: 409 });
  await assert.rejects(api.uploadBackend('engine', 'cpu', largeArchive, true), /chunk rejected/);
  assert.equal(requests.length - before, 2, 'Server error must stop subsequent chunks');

  const aborted = new AbortController();
  aborted.abort();
  before = requests.length;
  await assert.rejects(api.uploadBackend('engine', 'cpu', largeArchive, true, aborted.signal), { name: 'AbortError' });
  assert.equal(requests.length, before);
  const midUploadAbort = new AbortController();
  response = () => {
    midUploadAbort.abort();
    return Response.json({ status: 'uploading', received: chunkSize, total: largeArchive.size }, { status: 202 });
  };
  await assert.rejects(api.uploadBackend('engine', 'cpu', largeArchive, true, midUploadAbort.signal), { name: 'AbortError' });
  assert.equal(requests.length - before, 1, 'Cancellation must stop subsequent chunks');

  response = Response.json({ error: 'boundary reached server' }, { status: 400 });
  await assert.rejects(api.uploadBackend('engine', 'cpu', {
    ...largeArchive,
    size: 4 * 1024 ** 3,
  }, true), /boundary reached server/);
  before = requests.length;
  await assert.rejects(api.downloadBackendArchive('engine', 'cpu', 'https://example.com/asset', false), /Confirm/);
  for (const url of ['file:///tmp/engine.tar.gz', 'ftp://example.com/asset', 'http://', '', 'https://host/a b']) {
    assert.equal(apiModule.exports.isBackendArchiveUrl(url), false);
    await assert.rejects(api.downloadBackendArchive('engine', 'cpu', url, true), /HTTP or HTTPS/);
  }
  await assert.rejects(api.downloadBackendArchive('engine', 'cpu', 'http://localhost/asset', true, aborted.signal), { name: 'AbortError' });
  assert.equal(requests.length, before, 'Invalid or untrusted URL never contacts the server');
  api.setSessionAdminApiKey('admin-test-key');
  for (const source of ['http://127.0.0.1:8080/asset?token=a&b=c', 'https://example.com/download?id=123']) {
    response = Response.json({ status: 'success', recipe: 'engine', backend: 'cpu' });
    await api.downloadBackendArchive('engine', 'cpu', ` ${source} `, true, controller.signal);
    const { url, options } = requests.at(-1);
    const parsed = new URL(url);
    assert.equal(parsed.origin, 'https://offline-server.example:8443', 'Only the configured server fetches the archive');
    assert.equal(parsed.searchParams.get('url'), source);
    assert.equal(parsed.searchParams.get('recipe'), 'engine');
    assert.equal(parsed.searchParams.get('backend'), 'cpu');
    assert.equal(parsed.searchParams.has('upload_id'), false);
    assert.equal(parsed.searchParams.has('allow_unverified_release'), false);
    assert.equal(options.method, 'POST');
    assert.equal(options.body, undefined);
    assert.equal(options.signal, controller.signal);
    assert.equal(options.headers['X-Lemonade-Trust-Executable'], 'true');
    assert.equal(options.headers.Authorization, 'Bearer admin-test-key');
  }
  response = Response.json({ error: 'download refused in offline mode' }, { status: 400 });
  await assert.rejects(api.downloadBackendArchive('engine', 'cpu', 'http://localhost/asset', true), /offline mode/);
  for (const result of [{ status: 'success', recipe: 'wrong', backend: 'cpu' }, { status: 'uploading' }, null]) {
    response = Response.json(result);
    await assert.rejects(api.downloadBackendArchive('engine', 'cpu', 'https://example.com/asset', true), /Unexpected/);
  }
  assert.match(manager, /backend-upload-source/);
  assert.match(manager, /backend-upload-url/);
  before = requests.length;
  response = url => Number(new URL(url).searchParams.get('offset')) < chunkSize * 2
    ? Response.json({
      status: 'uploading',
      received: Number(new URL(url).searchParams.get('offset')) + chunkSize,
      total: largeArchive.size,
    }, { status: 202 })
    : Response.json({ status: 'success', recipe: 'strata', backend: 'rocm' });
  await api.uploadBackend('strata', 'rocm', largeArchive, true, undefined, true);
  assert.equal(requests.length - before, 3);
  for (const request of requests.slice(before)) {
    assert.equal(new URL(request.url).searchParams.get('allow_unverified_release'), 'true',
      'Every chunk must carry the same explicit risk acknowledgement');
  }
  response = Response.json({ status: 'success', recipe: 'strata', backend: 'rocm' });
  await api.downloadBackendArchive('strata', 'rocm', 'http://localhost/engine.tar.gz', true, undefined, true);
  assert.equal(new URL(requests.at(-1).url).searchParams.get('allow_unverified_release'), 'true');
  assert.match(manager, /api\.downloadBackendArchive\(recipe, backend, source\.url, true, undefined, allowUnverifiedRelease\)/);
  assert.match(manager, /Trust and install/);
  console.log('Backend uploads preserve sequential raw chunks, acknowledgements, limits, abort, trust, URL, admin auth, and in-use guards.');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
