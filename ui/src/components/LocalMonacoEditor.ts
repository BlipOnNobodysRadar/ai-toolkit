import Editor, { loader } from '@monaco-editor/react';

loader.config({ paths: { vs: '/monaco/vs' } });

export type { OnMount } from '@monaco-editor/react';
export default Editor;
