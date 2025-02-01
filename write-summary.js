import core from '@actions/core';

//const core = require('@actions/core');

console.log('Hello World');

async function run() {
    const summary = core.summary;
    summary.addRaw('<h1>My Job Summary</h1>');
    summary.addRaw('<p>This is a paragraph in the job summary.</p>');
    summary.addRaw('<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>');
    await summary.write();
}

run();

console.log('Hello World 2');
